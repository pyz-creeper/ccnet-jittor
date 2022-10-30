import jittor
from datasets.dataloaders import ADE20k
from networks.resnet import Resnet101
from networks.ccnet import CCHead



def test_single_gpu(model):
    # TODO inference each result on model (result in what form?)
    # TODO reshape the result so that it has the same shape with ann
    # TODO cal mIoU
    dataset = ADE20k(16,"../ADEChallengeData2016",train=False) # load val set!
    results = []
    for index, (img, ann) in enumerate(dataset):
        res = model(img)
        results.append(res)
    dataset.evaluate(results)
    pass


if __name__ == "__main__":
    model = Resnet101()
    head = CCHead()
    x = jittor.random([10,3,512,683])
    x = model(x)
    out = head(x)
    print(out.shape)
