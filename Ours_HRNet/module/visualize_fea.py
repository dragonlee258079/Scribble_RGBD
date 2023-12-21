import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2


def visualize_fea_channels(l1, l2, l3, l4, l5):
    plt.figure(figsize=(20, 20))
    l1_viz = l1.squeeze().cpu().data.numpy()
    for i, fea in enumerate(l1_viz):
        if i == 48:
            break
        plt.subplot(6, 8, i+1)
        plt.imshow(fea, cmap='gray')
        plt.axis("off")
    print(f"saving l1 features maps...")
    plt.savefig(f"visual_fea/channels/l1.png")
    plt.close()

    plt.figure(figsize=(20, 20))
    l2_viz = l2.squeeze().cpu().data.numpy()
    for i, fea in enumerate(l2_viz):
        plt.subplot(6, 8, i+1)
        plt.imshow(fea, cmap='gray')
        plt.axis("off")
    print(f"saving l2 features maps...")
    plt.savefig(f"visual_fea/channels/l2.png")
    plt.close()

    plt.figure(figsize=(20, 20))
    l3_viz = l3.squeeze().cpu().data.numpy()
    for i, fea in enumerate(l3_viz):
        plt.subplot(6, 8, i+1)
        plt.imshow(fea, cmap='gray')
        plt.axis("off")
    print(f"saving l3 features maps...")
    plt.savefig(f"visual_fea/channels/l3.png")
    plt.close()

    plt.figure(figsize=(20, 20))
    l4_viz = l4.squeeze().cpu().data.numpy()
    for i, fea in enumerate(l4_viz):
        plt.subplot(6, 8, i+1)
        plt.imshow(fea, cmap='gray')
        plt.axis("off")
    print(f"saving l4 features maps...")
    plt.savefig(f"visual_fea/channels/l4.png")
    plt.close()

    plt.figure(figsize=(20, 20))
    l5_viz = l5.squeeze().cpu().data.numpy()
    for i, fea in enumerate(l5_viz):
        plt.subplot(6, 8, i+1)
        plt.imshow(fea, cmap='gray')
        plt.axis("off")
    print(f"saving l5 features maps...")
    plt.savefig(f"visual_fea/channels/l5.png")
    plt.close()


def visualize_fea_mean(l1, l2, l3, l4, l5):
    plt.figure()
    l1 = torch.mean(l1.squeeze(), dim=0).cpu().data.numpy()
    plt.imshow(l1, cmap='gray')
    plt.axis("off")
    print(f"saving l1 mean features maps...")
    plt.savefig(f"visual_fea/mean/l1.png")
    plt.close()

    plt.figure()
    l2 = torch.mean(l2.squeeze(), dim=0).cpu().data.numpy()
    plt.imshow(l2, cmap='gray')
    plt.axis("off")
    print(f"saving l2 mean features maps...")
    plt.savefig(f"visual_fea/mean/l2.png")
    plt.close()

    plt.figure()
    l3 = torch.mean(l3.squeeze(), dim=0).cpu().data.numpy()
    plt.imshow(l3, cmap='gray')
    plt.axis("off")
    print(f"saving l3 mean features maps...")
    plt.savefig(f"visual_fea/mean/l3.png")
    plt.close()

    plt.figure()
    l4 = torch.mean(l4.squeeze(), dim=0).cpu().data.numpy()
    plt.imshow(l4, cmap='gray')
    plt.axis("off")
    print(f"saving l4 mean features maps...")
    plt.savefig(f"visual_fea/mean/l4.png")
    plt.close()

    plt.figure()
    l5 = torch.mean(l5.squeeze(), dim=0).cpu().data.numpy()
    plt.imshow(l5, cmap='gray')
    plt.axis("off")
    print(f"saving l5 mean features maps...")
    plt.savefig(f"visual_fea/mean/l5.png")
    plt.close()


def visualize_x_channels(x):
    plt.figure(figsize=(20, 20))
    x0_viz = x[0].squeeze().cpu().data.numpy()
    for i, fea in enumerate(x0_viz):
        plt.subplot(6, 8, i+1)
        plt.imshow(fea, cmap='gray')
        plt.axis("off")
    print(f"saving x0 features maps...")
    plt.savefig(f"visual_fea/channels/x0.png")
    plt.close()

    plt.figure(figsize=(20, 20))
    x1_viz = x[1].squeeze().cpu().data.numpy()
    for i, fea in enumerate(x1_viz):
        if i == 48:
            break
        plt.subplot(6, 8, i+1)
        plt.imshow(fea, cmap='gray')
        plt.axis("off")
    print(f"saving x1 features maps...")
    plt.savefig(f"visual_fea/channels/x1.png")
    plt.close()

    plt.figure(figsize=(20, 20))
    x2_viz = x[2].squeeze().cpu().data.numpy()
    for i, fea in enumerate(x2_viz):
        if i == 48:
            break
        plt.subplot(6, 8, i+1)
        plt.imshow(fea, cmap='gray')
        plt.axis("off")
    print(f"saving x2 features maps...")
    plt.savefig(f"visual_fea/channels/x2.png")
    plt.close()

    plt.figure(figsize=(20, 20))
    x3_viz = x[3].squeeze().cpu().data.numpy()
    for i, fea in enumerate(x3_viz):
        if i == 48:
            break
        plt.subplot(6, 8, i+1)
        plt.imshow(fea, cmap='gray')
        plt.axis("off")
    print(f"saving x3 features maps...")
    plt.savefig(f"visual_fea/channels/x3.png")
    plt.close()


def visualize_x_mean(x):
    plt.figure()
    x0 = torch.mean(x[0].squeeze(), dim=0).cpu().data.numpy()
    plt.imshow(x0, cmap='gray')
    plt.axis("off")
    print(f"saving x0 mean features maps...")
    plt.savefig(f"visual_fea/mean/x0.png")
    plt.close()

    plt.figure()
    x1 = torch.mean(x[1].squeeze(), dim=0).cpu().data.numpy()
    plt.imshow(x1, cmap='gray')
    plt.axis("off")
    print(f"saving x1 mean features maps...")
    plt.savefig(f"visual_fea/mean/x1.png")
    plt.close()

    plt.figure()
    x2 = torch.mean(x[2].squeeze(), dim=0).cpu().data.numpy()
    plt.imshow(x2, cmap='gray')
    plt.axis("off")
    print(f"saving x2 mean features maps...")
    plt.savefig(f"visual_fea/mean/x2.png")
    plt.close()

    plt.figure()
    x3 = torch.mean(x[3].squeeze(), dim=0).cpu().data.numpy()
    plt.imshow(x3, cmap='gray')
    plt.axis("off")
    print(f"saving x3 mean features maps...")
    plt.savefig(f"visual_fea/mean/x3.png")
    plt.close()

