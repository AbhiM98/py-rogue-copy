"""Prep uav images for inference."""
import cv2
import numpy as np


def clip_square_tiles_from_image(img, num_tiles=4) -> list:
    """
    Take in an image and preserve its aspect ratio by clipping it into squares.

    May trim some off the edges.
    """
    # get the image dimensions
    height, width, channels = img.shape
    print(f"Image shape: {img.shape}")

    # get the min dimension
    min_dim = min(height, width)

    # get the tile size
    tile_size = 2 * min_dim // num_tiles

    # get the center
    center_x = width // 2
    center_y = height // 2

    tile_coords = [
        (
            center_x + i * tile_size,
            center_y + j * tile_size,
            center_x + (i + 1) * tile_size,
            center_y + (j + 1) * tile_size,
        )
        for j in range(-num_tiles // 4, num_tiles // 4)
        for i in range(-num_tiles // 4, num_tiles // 4)
    ]

    print(*tile_coords, sep="\n")

    # get the tiles
    tiles = []
    for x1, y1, x2, y2 in tile_coords:
        tiles.append(img[y1:y2, x1:x2])

    return tiles


def stitch_tiles(tiles: list) -> np.ndarray:
    """
    Stiches the tiles back together.

    in the order
    1 2 3 ... n
    n+1 n+2 ...
    .
    .
    .
    m m+1 ... m+n
    """
    stitch_size = int(np.sqrt(len(tiles)))
    print(f"Stitch size: {stitch_size}")

    new_width = stitch_size * tiles[0].shape[1]
    new_height = stitch_size * tiles[0].shape[0]

    img = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    for i in range(stitch_size):
        for j in range(stitch_size):
            img[
                i * tiles[0].shape[0] : (i + 1) * tiles[0].shape[0],
                j * tiles[0].shape[1] : (j + 1) * tiles[0].shape[1],
                :,
            ] = tiles[i * stitch_size + j]

    return img


if __name__ == "__main__":
    # segments
    # segments = [
    #     KeySegment(0, "2023-dev-sandbox"),
    #     KeySegment(1, "uav_test"),
    #     KeySegment(2, is_file=True, incomplete_name="JPG"),
    # ]

    # # get_matching_s3_mirror_paths
    # img_mps = get_matching_s3_mirror_paths(segments)
    # print(f"Found {len(img_mps)} images.")

    # # crop_square_from_img_center
    # # download to mirror
    # img_mps[0].download_to_mirror()
    # print(img_mps[0].local_path)
    # img = cv2.imread(str(img_mps[0].local_path))

    # preproc_img = crop_square_from_img_center(img, 4096, rotate=False, ccw=False)
    # img_tiles = clip_square_tiles_from_image(img, num_tiles=4)

    # # write the tiles to a new folder
    # for i, tile in enumerate(img_tiles):
    #     output_path = img_mps[0].local_path.parent / f"cropped/cropped_uav_{i}.png"
    #     print(output_path)
    #     new_mp = MirrorPath.from_local_path(output_path)
    #     cv2.imwrite(str(output_path), tile)
    #     new_mp.upload_from_mirror(overwrite=True)

    # stitch the tiles
    tiles = [
        "/mnt/c/Users/nschroeder/Downloads/Rogues/cropped_uav_0.png",
        "/mnt/c/Users/nschroeder/Downloads/Rogues/cropped_uav_1.png",
        "/mnt/c/Users/nschroeder/Downloads/Rogues/cropped_uav_2.png",
        "/mnt/c/Users/nschroeder/Downloads/Rogues/cropped_uav_3.png",
    ]

    tiles = [cv2.imread(tile) for tile in tiles]
    composite = stitch_tiles(tiles)
    cv2.imwrite("/mnt/c/Users/nschroeder/Downloads/Rogues/composite.png", composite)
