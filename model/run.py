from torchvision import transforms
import image_classification as classifier

data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def main(image, model, labels):

    img = classifier.load_image_from_url(image, data_transform)
    net = classifier.load_resnet18_from_file(model)
    label_dict = classifier.load_labels(labels)
    result = classifier.predict(net, img)
    result = result.numpy()
    result = result[0]

    idx = result.argsort()
    idx = idx[::-1]
    num_selected = 5
    idx = idx[:num_selected]

    output = []
    for i in idx:
        prob = result[i]
        label = label_dict[i]
        output.append([label, str(prob)])

    return output
