import torch
import torchvision.transforms as TVT
import torchvision.transforms.functional as TF

def tensor_transforms3D(image, label=None, mask=None):
    slice, height, width = image.size()
    if torch.rand(1) > 0.2:
        angle = TVT.RandomRotation.get_params(degrees=(-10, 10))
        image = TF.rotate(image, angle)
        if label is not None:
            label = TF.rotate(label, angle)
        if mask is not None:
            mask = TF.rotate(mask, angle)

    if torch.rand(1) > 0.2:
        shift_max = 0.125
        dx_max = shift_max * width
        dy_max = shift_max * height
        translations = (torch.round(torch.FloatTensor(1).uniform_(-dx_max, dx_max)),
                        torch.round(torch.FloatTensor(1).uniform_(-dy_max, dy_max)))
        t_angle = 0
        t_scale = 1.0
        t_shear = 0.0
        image = TF.affine(image, t_angle, translations, t_scale, t_shear)
        if label is not None:
            label = TF.affine(label, t_angle, translations, t_scale, t_shear)
        if mask is not None:
            mask = TF.affine(mask, t_angle, translations, t_scale, t_shear)

    if torch.rand(1) > 0.2:
        scale = torch.FloatTensor(1).uniform_(0.8, 1.2)
        width_scaled = torch.round(width * scale).int()
        height_scaled = torch.round(height * scale).int()
        image = TF.resize(image, size=(height_scaled, width_scaled))
        if label is not None:
            label = TF.resize(label, size=(height_scaled, width_scaled))
        if mask is not None:
            mask = TF.resize(mask, size=(height_scaled, width_scaled))

    cur_slice, cur_height, cur_width = image.size()
    if width >= cur_width and height >= cur_height:
        pad_wid = width - cur_width
        left_pad_wid  = pad_wid // 2
        right_pad_wid = pad_wid - left_pad_wid
        pad_hei = height - cur_height
        top_pad_hei  = pad_hei // 2
        bottom_pad_hei = pad_hei - top_pad_hei
        image = TF.pad(image, padding=(left_pad_wid, top_pad_hei, right_pad_wid, bottom_pad_hei))
        if label is not None:
            label = TF.pad(label, padding=(left_pad_wid, top_pad_hei, right_pad_wid, bottom_pad_hei))
        if mask is not None:
            mask = TF.pad(mask, padding=(left_pad_wid, top_pad_hei, right_pad_wid, bottom_pad_hei))
    elif width >= cur_width and height < cur_height:
        pad_wid = width - cur_width
        left_pad_wid  = pad_wid // 2
        right_pad_wid = pad_wid - left_pad_wid
        image = TF.pad(image, padding=(left_pad_wid, 0, right_pad_wid, 0))
        if label is not None:
            label = TF.pad(label, padding=(left_pad_wid, 0, right_pad_wid, 0))
        if mask is not None:
            mask = TF.pad(mask, padding=(left_pad_wid, 0, right_pad_wid, 0))

        # Center cropping
        crop_hei = cur_height - height
        top_crop_hei = crop_hei // 2
        image = TF.crop(image, top_crop_hei, 0, height, width)
        if label is not None:
            label = TF.crop(label, top_crop_hei, 0, height, width)
        if mask is not None:
            mask = TF.crop(mask, top_crop_hei, 0, height, width)
    elif width < cur_width and height >= cur_height:
        pad_hei = height - cur_height
        top_pad_hei  = pad_hei // 2
        bottom_pad_hei = pad_hei - top_pad_hei
        image = TF.pad(image, padding=(0, top_pad_hei, 0, bottom_pad_hei))
        if label is not None:
            label = TF.pad(label, padding=(0, top_pad_hei, 0, bottom_pad_hei))
        if mask is not None:
            mask = TF.pad(mask, padding=(0, top_pad_hei, 0, bottom_pad_hei))

        crop_wid = cur_width - width
        left_crop_wid = crop_wid // 2
        image = TF.crop(image, 0, left_crop_wid, height, width)
        if label is not None:
            label = TF.crop(label, 0, left_crop_wid, height, width)
        if mask is not None:
            mask = TF.crop(mask, 0, left_crop_wid, height, width)
    elif width < cur_width and height < cur_height:
        crop_wid = cur_width - width
        left_crop_wid = crop_wid // 2
        crop_hei = cur_height - height
        top_crop_hei = crop_hei // 2
        image = TF.crop(image, top_crop_hei, left_crop_wid, height, width)
        if label is not None:
            label = TF.crop(label, top_crop_hei, left_crop_wid, height, width)
        if mask is not None:
            mask = TF.crop(mask, top_crop_hei, left_crop_wid, height, width)
    return image, label, mask
