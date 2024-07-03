import torch
import pdb

def extract_feature(model, loader):
    features = torch.FloatTensor()
    label = torch.FloatTensor()
    for (inputs, labels) in loader:

        input_img = inputs.to('cuda')
        outputs = model(input_img)
        f1 = outputs[0].data.cpu()

        # flip
        inputs = inputs.index_select(3, torch.arange(inputs.size(3) - 1, -1, -1))
        input_img = inputs.to('cuda')
        outputs = model(input_img)
        f2 = outputs[0].data.cpu()
        ff = f1 + f2

        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        features = torch.cat((features, ff), 0)
        label = torch.cat((label, labels), 0)
        
        p1 = outputs[-3].data.cpu()
        p2 = outputs[-2].data.cpu()
        p3 = outputs[-1].data.cpu()


    return features, label, p1, p2, p3

def extract_single_feature(model, loader):
    features = torch.FloatTensor()
    for (inputs, sizes) in loader:

        input_img = inputs.to('cuda')
        outputs = model(input_img)
        f1 = outputs[0].data.cpu()

        

        fnorm = torch.norm(f1, p=2, dim=1, keepdim=True)
        f1 = f1.div(fnorm.expand_as(f1))
        features = torch.cat((features, f1), 0)
        size = sizes
        
        p1 = outputs[-3].data.cpu()
        p2 = outputs[-2].data.cpu()
        p3 = outputs[-1].data.cpu()
        zg1 = outputs[-6].data.cpu()
        zp2 = outputs[-5].data.cpu()
        zp3 = outputs[-4].data.cpu()
        l_p1 = outputs[4].data.cpu()
        l_p2 = outputs[5].data.cpu()
        l_p3 = outputs[6].data.cpu()

    return features, size, p1, p2, p3, zg1, zp2, zp3, l_p1, l_p2, l_p3

def extract_feature_rank(model, loader):
    features = torch.FloatTensor()
    label = torch.FloatTensor()
    for (inputs, labels) in loader:

        input_img = inputs.to('cuda')
        outputs = model(input_img)
        f1 = outputs[0].data.cpu()

        # flip
        inputs = inputs.index_select(3, torch.arange(inputs.size(3) - 1, -1, -1))
        input_img = inputs.to('cuda')
        outputs = model(input_img)
        f2 = outputs[0].data.cpu()
        ff = f1 + f2

        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        features = torch.cat((features, ff), 0)
        
        labels = torch.tensor(labels)
        if label.numel() == 0:  # Check if label is empty
            label = labels
        else:
            label = torch.cat((label, labels), 0)

        

    return features, label

def extract_feature_dist(model, loader):
    features = torch.FloatTensor()
    label = torch.FloatTensor()
    for (inputs, labels) in loader:

        input_img = inputs.to('cuda')
        outputs = model(input_img)
        f1 = outputs[0].data.cpu()

        fnorm = torch.norm(f1, p=2, dim=1, keepdim=True)
        f1 = f1.div(fnorm.expand_as(f1))
        features = torch.cat((features, f1), 0)
        
        labels = torch.tensor(labels)
        if label.numel() == 0:  # Check if label is empty
            label = labels
        else:
            label = torch.cat((label, labels), 0)

        

    return features, label, fnorm