import argparse
import os

import cv2
import numpy as np
import torch
from tqdm import tqdm

from movinets import MoViNet
from movinets.config import _C


def resampled_video(input_path, target_fps=25):
    video_capture = cv2.VideoCapture(input_path)
    original_fps = video_capture.get(cv2.CAP_PROP_FPS)

    # Calculate the frame interval to achieve the target fps
    frame_interval = int(round(original_fps / target_fps))
    frame_count = 0

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            yield frame
        frame_count += 1

    video_capture.release()


def load_frame(frame, resize=False):
    # data = Image.open(frame)
    data = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # print(data.shape)  # (180, 320, 3)
    # assert data.shape[1] == 256
    # assert data.shape[0] == 340

    if resize:
        # data = data.resize((224, 224), Image.ANTIALIAS)
        data = cv2.resize(data, (224, 224), interpolation=cv2.INTER_AREA)

    data = data.astype(np.float32)
    data = (data * 2 / 255) - 1

    assert data.max() <= 1.0
    assert data.min() >= -1.0

    return data


def oversample_data(data):  # (39, 16, 224, 224, 2)  # Check twice
    data_flip = np.array(data[:, :, :, ::-1, :])

    data_1 = np.array(data[:, :, :224, :224, :])
    data_2 = np.array(data[:, :, :224, -224:, :])
    data_3 = np.array(data[:, :, 16:240, 58:282, :])
    data_4 = np.array(data[:, :, -224:, :224, :])
    data_5 = np.array(data[:, :, -224:, -224:, :])

    data_f_1 = np.array(data_flip[:, :, :224, :224, :])
    data_f_2 = np.array(data_flip[:, :, :224, -224:, :])
    data_f_3 = np.array(data_flip[:, :, 16:240, 58:282, :])
    data_f_4 = np.array(data_flip[:, :, -224:, :224, :])
    data_f_5 = np.array(data_flip[:, :, -224:, -224:, :])

    return [
        data_1,
        data_2,
        data_3,
        data_4,
        data_5,
        data_f_1,
        data_f_2,
        data_f_3,
        data_f_4,
        data_f_5,
    ]


def load_rgb_batch(rgb_files, frame_indices, resize=False):
    shape = (224, 224, 3) if resize else (256, 340, 3)
    batch_data = np.zeros(frame_indices.shape + shape, dtype=np.float32)

    for i in range(frame_indices.shape[0]):
        for j in range(frame_indices.shape[1]):
            batch_data[i, j, :, :, :] = load_frame(
                rgb_files[frame_indices[i][j]], resize
            )

    return batch_data


def run(
    sample_mode="oversample",
    frequency=16,
    input_dir="",
    output_dir="",
    batch_size=40,
):
    chunk_size = 16

    assert sample_mode in ["oversample", "center_crop", "resize"]

    model = MoViNet(_C.MODEL.MoViNetA2, causal=False, pretrained=True)
    model.classifier = torch.nn.Identity()  # type: ignore
    model.cuda()
    model.eval()

    torch.cuda.empty_cache()
    # data = torch.rand((1, 3, 16, 224, 224))

    def forward_batch(b_data):
        b_data = b_data.transpose([0, 4, 1, 2, 3])
        b_data = torch.from_numpy(b_data)  # b,c,t,h,w  # 40x3x16x224x224

        with torch.no_grad():
            model.clean_activation_buffers()
            b_data_cuda = b_data.cuda()
            b_features = model(b_data_cuda)

        b_features = b_features.data.cpu().numpy()
        return b_features

    video_names = [
        name
        for name in os.listdir(input_dir)
        if name.endswith("mp4") or name.endswith("avi")
    ]
    os.makedirs(output_dir, exist_ok=True)

    for video_name in tqdm(video_names):
        # save_file = '{}-{}.npz'.format(video_name, mode)
        save_file = "{}.npy".format(video_name)
        if save_file in os.listdir(output_dir):
            continue

        video_path = os.path.join(input_dir, video_name)
        rgb_files = [frame for frame in resampled_video(video_path)]
        frame_cnt = len(rgb_files)

        # clipped_length = (frame_cnt // chunk_size) * chunk_size   # Cut frames
        # Cut frames
        assert frame_cnt > chunk_size
        clipped_length = frame_cnt - chunk_size
        # The start of last chunk
        clipped_length = (clipped_length // frequency) * frequency

        frame_indices = []  # Frames to chunks
        for i in range(clipped_length // frequency + 1):
            frame_indices.append(
                [j for j in range(i * frequency, i * frequency + chunk_size)]
            )

        frame_indices = np.array(frame_indices)

        # frame_indices = np.reshape(frame_indices, (-1, 16)) # Frames to chunks
        chunk_num = frame_indices.shape[0]

        batch_num = int(np.ceil(chunk_num / batch_size))  # Chunks to batches
        frame_indices = np.array_split(frame_indices, batch_num, axis=0)

        if sample_mode == "oversample":
            full_features = [[] for i in range(10)]
        else:
            full_features = [[]]

        for batch_id in range(batch_num):
            require_resize = sample_mode == "resize"

            batch_data = load_rgb_batch(
                rgb_files, frame_indices[batch_id], require_resize
            )

            if sample_mode == "oversample":
                batch_data_ten_crop = oversample_data(batch_data)

                for i in range(10):
                    # print(batch_data_ten_crop[i].shape)  # (38, 16, 224, 224, 3)
                    assert batch_data_ten_crop[i].shape[-2] == 224
                    assert batch_data_ten_crop[i].shape[-3] == 224
                    full_features[i].append(forward_batch(batch_data_ten_crop[i]))

            else:
                if sample_mode == "center_crop":
                    # Centrer Crop  (39, 16, 224, 224, 2)
                    batch_data = batch_data[:, :, 16:240, 58:282, :]

                assert batch_data.shape[-2] == 224
                assert batch_data.shape[-3] == 224
                full_features[0].append(forward_batch(batch_data))

        full_features = [np.concatenate(i, axis=0) for i in full_features]
        full_features = [np.expand_dims(i, axis=0) for i in full_features]
        full_features = np.concatenate(full_features, axis=0)

        # np.savez(os.path.join(output_dir, save_file),
        #     feature=full_features,
        #     frame_cnt=frame_cnt,
        #     video_name=video_name)
        np.save(os.path.join(output_dir, save_file), full_features)

        print(
            f"{video_name} done: {frame_cnt} / {clipped_length}, {full_features.shape}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--batch_size", type=int, default=40)
    parser.add_argument("--sample_mode", type=str)
    parser.add_argument("--frequency", type=int, default=16)

    args = parser.parse_args()

    run(
        sample_mode=args.sample_mode,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        frequency=args.frequency,
    )
