from flask_restful import Resource, reqparse
from werkzeug.datastructures import FileStorage
from flask import current_app
import torch
from utils import global_model


class ResImageSearch(Resource):
    """restful api for image similarity

    Args:
        Resource (_type_): _description_
    """

    def post(self):
        """post image to search

        Returns:
            res:json
                possibility: list
                    id: int
                    similarity: float
                    url0: str
                    url1: str
        """
        parser = reqparse.RequestParser()
        parser.add_argument('file', required=True, type=FileStorage,
                            help='file is required', location='files')
        data = parser.parse_args()

        res = dict()
        res["possibility"] = []

        # 获取目标类别和目标类别的概率

        results = global_model.image_matcher.classify_and_search_similar_images(
            data.get('file'))
        torch.cuda.empty_cache()
        for result in results:
            if result[1] < 0.3:
                continue
            id = result[0]
            similarity = format(result[1], '.4f')
            url0 = current_app.config["IMAGE_BASE_URL"] + '/' + result[2][0]
            url1 = current_app.config["IMAGE_BASE_URL"] + '/' + result[2][1]

            dic = {"id": id, "similarity": similarity,
                   "url0": url0, "url1": url1}
            res["possibility"].append(dic)

        return res
