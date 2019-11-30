from data_conversion import parse_csv, Converter, make_numpy_dataset

def create_images():
    drum, drumnbass = make_numpy_dataset(img_size=(256, 256), limit=500)
    pass


if __name__ == "__main__":
    create_images()