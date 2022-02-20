import torch
from torch.autograd import Variable
from datasets import PoseDataset
from NeuralNetworks.model_decompose_nlblockDecoder import Model
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import argparse
import os
import time

torch.backends.cudnn.benchmark = True

# All command line parameters for the project
parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name', default='try', help="name of model")
parser.add_argument('-dr', '--root', default='/content/drive/MyDrive/DatasetCleanBBC', help="data root")
parser.add_argument('-l', '--lr', type=float, default=5e-4, help="learning rate")
parser.add_argument('-l1', '--lambda1', type=float, default=5, help="learning rate")
parser.add_argument('-l2', '--lambda2', type=float, default=1, help="learning rate")
parser.add_argument('-l3', '--lambda3', type=float, default=0.5, help="learning rate")
parser.add_argument('-r', '--recurrent', type=int, default=3, help='number of recurrent')
parser.add_argument('-d', '--dataset', default='clean_bbc', help='dataset')
parser.add_argument('-R', '--roi', action='store_true', help="use ROI split")
parser.add_argument('-f', '--finetune', action='store_true', help="use pretrained model")
parser.add_argument('-m', '--model', help="finetune model path")
parser.add_argument('-di', '--dilate', action='store_true', help='use dilated model')
parser.add_argument('-dp', '--dual_path', action='store_true', help='use two stream E')
parser.add_argument('-nl', '--non_local', action='store_true', help='use non local block')
parser.add_argument('-gan', '--use_gan', action='store_true', help='use non local block')
parser.add_argument('-sum', '--sum', action='store_true', help='use non local block')
parser.add_argument('-DVG', '--DVG', action='store_true', help='use non local block')

parser.add_argument('--local_rank', type=int, help='only used for distributed training')
parser.add_argument('--num_gpus', type=int, default=4)

args = parser.parse_args()

# Using GPU for processing
device = 'cuda'
dataset = args.dataset


# Function for normalizing the input image
def norm_ip(img, minimum, maximum):
    img.clamp_(min=minimum, max=maximum)
    img.add_(-minimum).div_(maximum - minimum)


# creating that model
model_name = f"{args.name}_lr{args.lr}_recurrent{args.recurrent}_lambda1{args.lambda1}_lambda2{args.lambda2}"

if not args.roi:
    model_name += '_withoutROI'

writer = SummaryWriter(os.path.join('./logs/', args.dataset, model_name))
root = args.root

root = os.path.join(root, dataset)

# This is preprocessing of data
train_data = PoseDataset(root, mode='train')
test_data = PoseDataset(root, mode='test')

image_loader = DataLoader(train_data,
                          num_workers=4,
                          batch_size=12,
                          shuffle=True,
                          pin_memory=True)

test_loader = DataLoader(test_data,
                         num_workers=2,
                         batch_size=5,
                         shuffle=True,
                         pin_memory=True)

if args.dataset == 'market1501':
    pose_channels = 18
elif args.dataset == 'Penn_Action':
    pose_channels = 13
else:
    pose_channels = 7

perceptual_mapping = {
    '2': 'conv1_2',
    '7': 'conv2_2',
    '12': 'conv3_2',
    '21': 'conv4_2',
    '30': 'conv5_2'
}

if args.roi:
    ROI_size = [(30, 30), (30, 30), (15, 15), (7, 7), (4, 4)]
else:
    ROI_size = None


# this is training
model = Model(pose_channels, 3, recurrent=args.recurrent,
              use_nonlocal=args.non_local, dataset=args.dataset,
              use_gan=args.use_gan, lr=args.lr,
              lambda1=args.lambda1, lambda2=args.lambda2, lambda3=args.lambda3,
              mapping=perceptual_mapping, roi_size=ROI_size, DVG=args.DVG, dual_path=args.dual_path).to('cuda')

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

if args.finetune:
    model.module.load_state_dict(torch.load(args.model))
model.train()

lr = args.lr

model_dir = os.path.join('./models', model_name)

start_e = 0

start = time.time()

for epoch in range(start_e, start_e + 200):
    cur_time = time.time()

    for iter, (pose_imgs, target_imgs, human_imgs, background_imgs, masks, target_masks, roi_bbox) in enumerate(
            image_loader):
        idx_tensor = torch.FloatTensor(list(range(2 * pose_imgs.shape[0]))).view(pose_imgs.shape[0] * 2).unsqueeze(-1)
        model = model.to(device)
        if torch.cuda.is_available():
            pose_imgs, target_imgs, human_imgs, background_imgs, masks, target_masks = pose_imgs.to(
                device), target_imgs.to(device), human_imgs.to(device), background_imgs.to(device), masks.to(
                device), target_masks.to(device)
            roi_bbox = torch.cat([roi_bbox, roi_bbox], 0)
            roi_bbox = Variable(torch.cat([idx_tensor, roi_bbox], 1), requires_grad=False).to(device)
        if args.DVG or not args.roi:
            roi_bbox = None

        perc_loss = []
        masked_perc_loss = []
        recons_loss = []
        extra_loss = []
        ssim = []
        src_imgs = human_imgs + background_imgs

        pred, recons_loss, perc_loss, mask_perc_loss, gan_loss, d_loss = model.optimize(pose_imgs, human_imgs,
                                                                                        background_imgs, masks,
                                                                                        target_imgs, roi_bbox)

        if iter % 500 == 0:
            log = 'Epoch %d Iter %d ReconsLoss %.4f perceptual loss %.4f' % (
                epoch + 1, iter + 1, recons_loss.item(), perc_loss.item())
            if args.roi:
                log += ' masked perceptual loss %.4f' % (mask_perc_loss.item())
            if args.use_gan:
                log += ' netD loss %.4f netG loss %.4f' % (d_loss.item(), gan_loss.item())
            log += ' elapse time {:.2f}s'.format(time.time() - cur_time)
            cur_time = time.time()
            print(log)
    torch.cuda.empty_cache()

    if (epoch + 1) % 1 == 0:
        pred = pred[-1]
        writer.add_image("Train Pred Images", make_grid(pred[:5].data, normalize=True, scale_each=True), epoch + 1)
        writer.add_image("Train Target Images", make_grid(target_imgs[:5].data, normalize=True, scale_each=True),
                         epoch + 1)
        writer.add_image("Train Src Images", make_grid(src_imgs[:5].data, normalize=True, scale_each=True), epoch + 1)
        model.eval()
        with torch.no_grad():
            print('Testing...')
            for _, (pose_imgs, target_imgs, human_imgs, background_imgs, masks, target_masks, roi_bbox, src_name,
                    target_name) in enumerate(test_loader):
                if torch.cuda.is_available():
                    pose_imgs, target_imgs, human_imgs, background_imgs, masks = pose_imgs.to(device), target_imgs.to(
                        device), human_imgs.to(device), background_imgs.to(device), masks.to(device)

                src_imgs = human_imgs + background_imgs
                pred = model((pose_imgs, human_imgs, background_imgs, masks))
                pred = pred[-1]

                writer.add_image("Test Pred Images", make_grid(pred.data, normalize=True, scale_each=True), epoch + 1)
                writer.add_image("Test Target Images", make_grid(target_imgs.data, normalize=True, scale_each=True),
                                 epoch + 1)
                writer.add_image("Test Src Images", make_grid(src_imgs.data, normalize=True, scale_each=True),
                                 epoch + 1)
                break
        if not os.path.isdir(os.path.join('./models', model_name)):
            os.mkdir(os.path.join('./models', model_name))
        if epoch % 20 == 0:
            torch.save(model.state_dict(),
                       os.path.join('./models', model_name, '_'.join([args.dataset, '%d.pt' % (epoch + 1)])))
        torch.cuda.empty_cache()
    model.train()
    torch.cuda.empty_cache()

torch.save(model.state_dict(), os.path.join('./models', model_name, '_'.join([args.dataset, '%d.pt' % (epoch + 1)])))
print("total time:" + str(time.time() - start))
if 1:
    writer.close()
