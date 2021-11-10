import torch


def get_large_small_rois(rois):
    ### enhanced rois
    ctr_x = ((rois[:, 1] + rois[:, 3]) * 0.5).view(-1, 1)
    ctr_y = ((rois[:, 2] + rois[:, 4]) * 0.5).view(-1, 1)
    rw = (rois[:, 3] - rois[:, 1] + 1.0).view(-1, 1)
    rh = (rois[:, 4] - rois[:, 2] + 1.0).view(-1, 1)

    zero_c = torch.tensor(0.1).cuda()

    large_rate = 2.0
    small_rate = 0.5
    large_w = rw * large_rate
    large_h = rh * large_rate

    small_w = rw * small_rate
    small_h = rh * small_rate
    
    # area shrinks half, using 1.414 as rate
    large_rois = torch.cat(
        (rois[:, 0].view(-1, 1),
         torch.max(ctr_x - large_w * 0.5, zero_c),
         torch.max(ctr_y - large_h * 0.5, zero_c),
         ctr_x + large_w * 0.5,
         ctr_y + large_h * 0.5), dim=-1)
    small_rois = torch.cat(
        (rois[:, 0].view(-1, 1),
         torch.max(ctr_x - small_w * 0.5, zero_c),
         torch.max(ctr_y - small_h * 0.5, zero_c),
         ctr_x + small_w * 0.5,
         ctr_y + small_h * 0.5), dim=-1)

    return large_rois, small_rois


def get_adaptive_scale_rois(rois, facs):
    ### enhanced rois
    ctr_x = ((rois[:, 1] + rois[:, 3]) * 0.5).view(-1, 1)
    ctr_y = ((rois[:, 2] + rois[:, 4]) * 0.5).view(-1, 1)
    rw = (rois[:, 3] - rois[:, 1] + 1.0).view(-1, 1)
    rh = (rois[:, 4] - rois[:, 2] + 1.0).view(-1, 1)

    # zero_c = torch.tensor(0.1).cuda()
    zero_c = torch.ones_like(rw) * 0.1
    # device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    # zero_c = torch.tensor(0.1).to(device)

    h_rate = (rw / rh) * facs + 1.0
    w_rate = (rh / rw) * facs + 1.0
    large_h = rh * h_rate
    large_w = rw * w_rate
    
    # area shrinks half, using 1.414 as rate
    adaptive_h_rois = torch.cat(
        (rois[:, 0].view(-1, 1),
         torch.max(ctr_x - rw * 0.5, zero_c),
         torch.max(ctr_y - large_h * 0.5, zero_c),
         ctr_x + rw * 0.5,
         ctr_y + large_h * 0.5), dim=-1)
    adaptive_w_rois = torch.cat(
        (rois[:, 0].view(-1, 1),
         torch.max(ctr_x - large_w * 0.5, zero_c),
         # torch.max(ctr_y - rh * 0.5, zero_c),
         torch.max(ctr_y - large_h * 0.5, zero_c),
         ctr_x + large_w * 0.5,
         ctr_y + large_h * 0.5), dim=-1)
         # ctr_y + rh * 0.5), dim=-1)

    return adaptive_h_rois, adaptive_w_rois


def get_large_wh_rois(rois):
    ### enhanced rois
    ctr_x = ((rois[:, 1] + rois[:, 3]) * 0.5).view(-1, 1)
    ctr_y = ((rois[:, 2] + rois[:, 4]) * 0.5).view(-1, 1)
    rw = (rois[:, 3] - rois[:, 1] + 1.0).view(-1, 1)
    rh = (rois[:, 4] - rois[:, 2] + 1.0).view(-1, 1)

    # zero_c = torch.tensor(0.1).cuda()
    zero_c = torch.ones(1) * 0.1

    large_rate = 3
    large_w = rw * large_rate
    large_h = rh * large_rate
    
    # area shrinks half, using 1.414 as rate
    large_w_rois = torch.cat(
        (rois[:, 0].view(-1, 1),
         torch.max(ctr_x - large_w * 0.5, zero_c),
         torch.max(ctr_y - rh * 0.5, zero_c),
         ctr_x + large_w * 0.5,
         ctr_y + rh * 0.5), dim=-1)
    large_h_rois = torch.cat(
        (rois[:, 0].view(-1, 1),
         torch.max(ctr_x - rw * 0.5, zero_c),
         torch.max(ctr_y - large_h * 0.5, zero_c),
         ctr_x + rw * 0.5,
         ctr_y + large_h * 0.5), dim=-1)

    return large_w_rois, large_h_rois

def get_small_wh_rois(rois):
    ### enhanced rois
    ctr_x = ((rois[:, 1] + rois[:, 3]) * 0.5).view(-1, 1)
    ctr_y = ((rois[:, 2] + rois[:, 4]) * 0.5).view(-1, 1)
    rw = (rois[:, 3] - rois[:, 1] + 1.0).view(-1, 1)
    rh = (rois[:, 4] - rois[:, 2] + 1.0).view(-1, 1)

    zero_c = torch.tensor(0.1).cuda()

    small_rate = 0.33
    lw_w = rw
    lw_h = rh * small_rate
    
    lh_w = rw * small_rate
    lh_h = rh

    # small_rate = 0.707
    # small_w = rw * small_rate
    # small_h = rh * small_rate
    
    # area shrinks half, using 1.414 as rate
    small_w_rois = torch.cat(
        (rois[:, 0].view(-1, 1),
         torch.max(ctr_x - lw_w * 0.5, zero_c),
         torch.max(ctr_y - lw_h * 0.5, zero_c),
         ctr_x + lw_w * 0.5,
         ctr_y + lw_h * 0.5), dim=-1)
    small_h_rois = torch.cat(
        (rois[:, 0].view(-1, 1),
         torch.max(ctr_x - lh_w * 0.5, zero_c),
         torch.max(ctr_y - lw_h * 0.5, zero_c),
         ctr_x + lh_w * 0.5,
         ctr_y + lh_h * 0.5), dim=-1)

    return small_w_rois, small_h_rois


def get_boundary_rois(rois):
    ### enhanced rois
    ctr_x = ((rois[:, 1] + rois[:, 3]) * 0.5).view(-1, 1)
    ctr_y = ((rois[:, 2] + rois[:, 4]) * 0.5).view(-1, 1)
    rw = (rois[:, 3] - rois[:, 1] + 1.0).view(-1, 1)
    rh = (rois[:, 4] - rois[:, 2] + 1.0).view(-1, 1)

    zero_c = torch.tensor(0.1).cuda()

    small_rate = 0.5
    small_w = rw * small_rate
    small_h = rh * small_rate

    # top center (ctr_x, rois[:, 2). width keeps still, height * 0.5
    top_rois = torch.cat(
        (rois[:, 0].view(-1, 1),
         torch.max(ctr_x - small_w, zero_c),
         torch.max(rois[:, 2].view(-1, 1) - small_h * 0.5, zero_c),
         ctr_x + small_w,
         rois[:, 2].view(-1, 1) + small_h * 0.5), dim=-1)
    # bottom center (ctr_x, rois[:, 4). width keeps still, height * 0.5
    bottom_rois = torch.cat(
        (rois[:, 0].view(-1, 1),
         torch.max(ctr_x - small_w, zero_c),
         torch.max(rois[:, 4].view(-1, 1) - small_h * 0.5, zero_c),
         ctr_x + small_w,
         rois[:, 4].view(-1, 1) + small_h * 0.5), dim=-1)
    # left center (rois[:, 1], ctr_y. height keeps still, width * 0.5
    left_rois = torch.cat(
        (rois[:, 0].view(-1, 1),
         torch.max(rois[:, 1].view(-1, 1) - small_w * 0.5, zero_c),
         torch.max(ctr_y - small_h, zero_c),
         rois[:, 1].view(-1, 1) + small_w * 0.5,
         ctr_y + small_h), dim=-1)

    # right center (rois[:, 3], ctr_y. height keeps still, width * 0.5
    right_rois = torch.cat(
        (rois[:, 0].view(-1, 1),
         torch.max(rois[:, 3].view(-1, 1) - small_w * 0.5, zero_c),
         torch.max(ctr_y - small_h, zero_c),
         rois[:, 3].view(-1, 1) + small_w * 0.5,
         ctr_y + small_h), dim=-1)

    return top_rois, right_rois, bottom_rois, left_rois


def get_context_rois(rois):
    ### enhanced rois
    ctr_x = ((rois[:, 1] + rois[:, 3]) * 0.5).view(-1, 1)
    ctr_y = ((rois[:, 2] + rois[:, 4]) * 0.5).view(-1, 1)
    rw = (rois[:, 3] - rois[:, 1] + 1.0).view(-1, 1)
    rh = (rois[:, 4] - rois[:, 2] + 1.0).view(-1, 1)
    
    zero_c = torch.ones_like(rw) * 0.1
    # zero_c = torch.tensor(0.1).cuda()
    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # zero_c = torch.tensor(0.1).to(device)

    wdh = rw / rh
    hdw = rh / rw
    wdh = torch.where(wdh>2.0, torch.ones_like(wdh) * 2.0, wdh)
    hdw = torch.where(hdw>2.0, torch.ones_like(hdw) * 2.0, hdw)
    
    h1_rate = torch.where(rh < rw, wdh, torch.zeros_like(rh)) + 1.0
    w1_rate = torch.where(rh < rw, torch.zeros_like(rh), hdw) + 1.0
    h2_rate = wdh + 1.0
    w2_rate = hdw + 1.0
    
    adaptive_h_rois = torch.cat(
        (rois[:, 0].view(-1, 1),
         torch.max(ctr_x - rw * w1_rate * 0.5, zero_c),
         torch.max(ctr_y - rh * h1_rate * 0.5, zero_c),
         ctr_x + rw * w1_rate * 0.5,
         ctr_y + rh * h1_rate * 0.5), dim=-1)
    adaptive_w_rois = torch.cat(
        (rois[:, 0].view(-1, 1),
         torch.max(ctr_x - rw * w2_rate * 0.5, zero_c),
         torch.max(ctr_y - rh * h2_rate * 0.5, zero_c),
         ctr_x + rw * w2_rate * 0.5,
         ctr_y + rh * h2_rate * 0.5), dim=-1)
    
    return adaptive_h_rois, adaptive_w_rois
