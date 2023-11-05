from torchvision import transforms, models
import torch
import torch.nn as nn
from PIL import Image


class Classifier:
    CLASSES = 102

    def __init__(self, checkpoint_path: str):
        self.model = self._load_model(checkpoint_path)
        self.model.eval()

    def _load_model(self, checkpoint_path: str):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(
            checkpoint_path, map_location=device)

        model = models.vgg16()  # no default weight
        model.classifier[-1].out_features = Classifier.CLASSES
        model.class_to_idx = checkpoint['class_to_idx']
        model.load_state_dict(checkpoint['state_dict'])

        for param in model.parameters():
            param.requires_grad = False

        return model

    def _process_image(self, image):
        img = Image.open(image).convert("RGB")
        adjust = transforms.Compose([transforms.Resize((224, 224)),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])

        img_tensor = adjust(img)

        return img_tensor

    """ 分类图片，默认返回top5的可能种类与其概率 """

    def classify(self, image, topk=5):
        processed_img = self._process_image(image)
        processed_img = torch.unsqueeze(processed_img, 0)  # 增加一维
        prob = nn.Softmax(dim=1)

        with torch.no_grad():
            output = self.model(processed_img)
            values, category = torch.topk(output.data, k=topk, dim=1)
            topk_prob = prob(values)

        idx_to_class = {v: k for k, v in self.model.class_to_idx.items()}

        topk_prob = topk_prob[0].numpy().tolist()  # 去掉一维，并转为list

        topk_class = []
        for x in range(topk):
            true_class = idx_to_class[int(category[0][x])]
            topk_class.append(int(true_class))

        return topk_class, topk_prob
