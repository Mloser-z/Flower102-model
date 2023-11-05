from utils.ImageMatcher import ImageMatcher


def global_init(config):
    global image_matcher

    image_matcher = ImageMatcher(config["MODEL_PATH"],
                                 config["IMAGE_DIR"] + '/')
