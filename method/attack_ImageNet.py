import numpy as np
import os
import gc
import random
import argparse
import torch
import torchvision

from diffusers import StableDiffusionPipeline, DDIMScheduler
from tqdm import tqdm
from advertorch.attacks import LinfPGDAttack

from utils import *
from pgd_latent_optimize_ImageNet import run_latent_space_optimization_imagenet
from attentionControl import AttentionControlEdit


def run_SDM(ref_image, image, label_id, diffusion_model, self_replace_steps=1., device=None, args=None):
    
    controller = AttentionControlEdit(args.diffusion_steps, self_replace_steps, args.res)
    
    image_hat = run_latent_space_optimization_imagenet(
        ref_image, image, diffusion_model, controller, label_id,
        res=args.res,
        num_inference_steps=args.diffusion_steps,
        guidance_scale=args.guidance,
        start_step=args.start_step,
        iterations=args.iterations,                                            
        args=args,
        device=device,
    )

    return image_hat


def generate_x_adv_pgd_attacked(x, y, classifier, pgd_conf):
    
    adversary = LinfPGDAttack(
        classifier,
        loss_fn=torch.nn.CrossEntropyLoss(reduction='sum'),
        eps=pgd_conf['eps'],
        nb_iter=pgd_conf['iter'],
        eps_iter=pgd_conf['alpha'],
        rand_init=False,
        targeted=True  # 改为目标攻击        
    )
    
    x_adv = adversary.perturb(x, y)
    
    return x_adv
    

def run_PiA_LaO(args=None):
    device = torch.device(f"cuda:{args.gpuNum}" if torch.cuda.is_available() else "cpu")

    loggingPath = os.path.join(args.logging_root, 'train_PiA-LaO.log')
    PGDLogger = get_logger(loggingPath, name=f'PiA-LaO_{args.wm_method}')
    
    savePath = args.save_images_root
    os.makedirs(savePath, exist_ok=True)
    
    pgd_conf = gen_pgd_confs(eps=args.eps, alpha=args.alpha, iter=args.iter, input_range=(0, 1))
    
    ckptPath = os.path.join(args.classifier_ckpt_root, f'ResNet50_{args.wm_method}.pth')
    print(ckptPath)
    classifier = ResNet50_BinaryClassifier()
    classifier.load_state_dict(torch.load(ckptPath, map_location='cpu'))
    classifier = classifier.to(device)
    classifier.eval()
    
    clean_data_path = args.clean_wm_images_root
    clean_dataset = CustomImageFolder(clean_data_path, args.data_cnt)
    org_dataset = CustomImageFolder(args.org_images, args.data_cnt)
    
    label_id = get_imagenet_label_id(args.imagenet_label_path)
    
    ldm_stable = StableDiffusionPipeline.from_pretrained(args.pretrained_diffusion_path).to(device)

    ldm_stable.scheduler = DDIMScheduler.from_config(ldm_stable.scheduler.config)
    
    clean_all_acc = 0
    adv_all_acc = 0
    
    for i in tqdm(range(args.data_cnt)):
        orgImage = org_dataset[i]
        org_image = preprocess(orgImage, res=args.res).to(device)
        
        cleanImage = clean_dataset[i]
        
        x = preprocess(cleanImage, res=args.res).to(device)  # (torch.Size([1, 3, 256, 256]), [0. , 1.])
        y = torch.tensor(0)[None].to(device)
            
        y_pred = classifier(x).argmax(1) # original prediction
        pred_clean_accuracy = (y_pred[0] == 1).sum().item()
        
        PGDLogger.info(f'True label: 1\tClean watermarked image predict: {y_pred[0]}')
        
        # 1- clean image latent optimize
        x_hat = run_SDM(org_image, x, label_id[i:i+1], ldm_stable, device=device, args=args)
        y_pred_hat = classifier(x_hat).argmax(1) 
        PGDLogger.info(f'Diff watermarked image predict: {y_pred_hat[0]}')
        
        # 2- pgd attack optimized image
        x_adv = generate_x_adv_pgd_attacked(x_hat, y, classifier, pgd_conf)
        y_pred_adv = classifier(x_adv).argmax(1)
        PGDLogger.info(f'PGD watermarked image predict: {y_pred_adv[0]}')
        
        # 3- attacked image latent optimize
        x_adv_hat = run_SDM(org_image, x_adv, label_id[i:i+1], ldm_stable, device=device, args=args)
        y_pred_adv_hat = classifier(x_adv_hat).argmax(1)
        pred_adv_accuracy = (y_pred_adv_hat[0] == 0).sum().item()
        PGDLogger.info(f'Diff_PGD watermarked image predict: {y_pred_adv_hat[0]}')
        
        clean_all_acc += pred_clean_accuracy
        adv_all_acc += pred_adv_accuracy
        
        # si(torch.cat([org_image, x, x_adv_hat, (x_adv_hat-x)*50], -1), savePath + f'/{i+1}.png')
        # os.makedirs(os.path.join(savePath, 'adv_hat'), exist_ok=True)
        # si(x_adv_hat, os.path.join(savePath, 'adv_hat', f'{i+1}.png'))

        si(x_adv_hat, savePath + f'/{i+1}.png')
    
    PGDLogger.info("Clean acc: {}%".format(clean_all_acc / args.data_cnt * 100))
    PGDLogger.info("Adv acc: {}%".format(adv_all_acc / args.data_cnt * 100))
        

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--wm_method', default='StrongMark_denoise_1', type=str,
                        # choices=["dwtDct", "dwtDctSvd", "rivaGan", "SSL", "stegaStamp", "ZoDiac", "ybbMark_size256"]
                        )
    parser.add_argument('--gpuNum', default=7, type=int)
    parser.add_argument('--org_images', default="../dataset/imagenet_compatible_org", type=str)
    parser.add_argument('--clean_wm_images_root', default="StrongMark_results/denoise_steps_1/wm_tmp", type=str)
    parser.add_argument('--data_cnt', type=int, default=100)
    parser.add_argument('--imagenet_label_path', default="../dataset/imagenet_compatible_labels.txt", type=str)
    parser.add_argument('--logging_root', default="StrongMark_results/denoise_steps_1/log", type=str)
    parser.add_argument('--save_images_root', default='StrongMark_results/denoise_steps_10/attacked/PiA-LaO', type=str)
    parser.add_argument('--classifier_ckpt_root', default='../classifier_ckpt/StrongMark', type=str)
    
    # PGD attack parameters
    parser.add_argument('--eps', default=16, type=int)
    parser.add_argument('--iter', default=10, type=int)
    parser.add_argument('--alpha', default=2, type=int)
    
    # Stable Diffusion parameters
    parser.add_argument('--pretrained_diffusion_path', default="HuggingFaceModels/stable-diffusion-2-1-base", type=str)
    parser.add_argument('--diffusion_steps', default=20, type=int, help='Total DDIM sampling steps')
    parser.add_argument('--start_step', default=15, type=int, help='Which DDIM step to start the attack')
    parser.add_argument('--iterations', default=15, type=int, help='Iterations of optimizing the adv_image')
    parser.add_argument('--res', default=256, type=int, help='Input image resized resolution')
    parser.add_argument('--guidance', default=2.5, type=float, help='guidance scale of diffusion models')
    parser.add_argument('--lpips_loss_weight', default=10, type=int, help='self attention loss weight factor')
    parser.add_argument('--ssim_loss_weight', default=10, type=int, help='self attention loss weight factor')
    parser.add_argument('--self_attn_loss_weight', default=10, type=int, help='self attention loss weight factor')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    run_PiA_LaO(args=args)
