import numpy as np
import os
from saliency_metric import cal_mae,cal_fm,cal_sm,cal_em,cal_wfm, cal_dice, cal_iou,cal_ber,cal_acc, HCEMeasure
from tqdm import tqdm
import cv2
from skimage.morphology import skeletonize
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image


class TestDataset(Dataset):
    def __init__(self, pred_root, gt_root):
        super().__init__()
        image_files = {os.path.splitext(f)[0] for f in os.listdir(pred_root)
                       if f.lower().endswith(('.png', '.jpg', '.bmp'))}
        gt_files = {os.path.splitext(f)[0] for f in os.listdir(gt_root)
                    if f.lower().endswith(('.png', '.jpg', '.bmp'))}
        self.img_list = list(image_files & gt_files)
        self.size = len(self.img_list)
        self.pred_root = pred_root
        self.gt_root = gt_root


    def __len__(self):
        return len(self.img_list)
    def __getitem__(self, idx):
        base_name = self.img_list[idx]

        pred_path = self.find_image_path(self.pred_root, base_name)
        gt_path = self.find_image_path(self.gt_root, base_name)

        pred = self.binary_loader(pred_path)
        gt = self.binary_loader(gt_path)

        pred = np.asarray(pred,np.float64)
        gt = np.asarray(gt,np.float64)

        return pred, gt, gt_path

    def find_image_path(self, root, base_name):
        for ext in ['.png', '.jpg', '.bmp']:
            path = os.path.join(root, base_name + ext)
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"No image found for {base_name} in {root}")

    @staticmethod
    def rgb_loader(path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    @staticmethod
    def binary_loader(path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')


diste1 = '/vip_media/haipeng/DIS5K/DIS-TE1/gt'
diste2 = '/vip_media/haipeng/DIS5K/DIS-TE2/gt'
diste3 = '/vip_media/haipeng/DIS5K/DIS-TE3/gt'
diste4 = '/vip_media/haipeng/DIS5K/DIS-TE4/gt'
#disvd = '/home/vanessa/code/DIS-main/DIS5K/DIS-VD/'

diste1 = os.path.join(diste1)
diste2 = os.path.join(diste2)
diste3 = os.path.join(diste3)
diste4 = os.path.join(diste4)

splits = {
    'DIS-TE1':diste1,
    'DIS-TE2':diste2,
    'DIS-TE3':diste3,
    'DIS-TE4':diste4,
    #'DIS-VD':disvd,
                }


pred = [
    '/home/haipeng/ICCV-Result/Model_80'
    ]


def evaluate_metrics(dir, test_datasets, metrics_to_compute):
    results = {}

    for d in dir:
        for name, root in test_datasets.items():
            pred_root = os.path.join(d, name)
            gt_root = root

            test_set = TestDataset(pred_root, gt_root)
            loader = DataLoader(test_set, batch_size=1, shuffle=False)

            # Initialize metrics
            metrics = {
                'mae': cal_mae(),
                'fm': cal_fm(test_set.size),
                'sm': cal_sm(),
                'em': cal_em(),
                'wfm': cal_wfm(),
                'm_dice': cal_dice(),
                'm_iou': cal_iou(),
                'ber': cal_ber(),
                'acc': cal_acc(),
                'hce': HCEMeasure()
            }

            for i, (pred, gt, gt_path) in enumerate(tqdm(loader)):
                gt = np.asarray(gt, np.float64)
                gt /= (gt.max() + 1e-8)
                gt[gt > 0.5] = 1
                gt[gt != 1] = 0
                res = pred
                res = np.array(res, np.float64)
                if res.max() == res.min():
                    res = res / 255
                else:
                    res = (res - res.min()) / (res.max() - res.min())

                gt = gt[0]
                res = res[0]

                ske_path = gt_path[0].replace("/gt/", "/ske/")
                if os.path.exists(ske_path):
                    ske_ary = cv2.imread(ske_path, cv2.IMREAD_GRAYSCALE)
                    ske_ary = ske_ary > 128
                else:
                    ske_ary = skeletonize(gt > 0.5)
                    ske_save_dir = os.path.join(*ske_path.split(os.sep)[:-1])
                    if ske_path[0] == os.sep:
                        ske_save_dir = os.sep + ske_save_dir
                    os.makedirs(ske_save_dir, exist_ok=True)
                    cv2.imwrite(ske_path, ske_ary.astype(np.uint8) * 255)

                # Update selected metrics
                if 'mae' in metrics_to_compute:
                    metrics['mae'].update(res, gt)
                if 'sm' in metrics_to_compute:
                    metrics['sm'].update(res, gt)
                if 'fm' in metrics_to_compute:
                    metrics['fm'].update(res, gt)
                if 'em' in metrics_to_compute:
                    metrics['em'].update(res, gt)
                if 'wfm' in metrics_to_compute:
                    metrics['wfm'].update(res, gt)
                if 'm_dice' in metrics_to_compute:
                    metrics['m_dice'].update(res, gt)
                if 'm_iou' in metrics_to_compute:
                    metrics['m_iou'].update(res, gt)
                if 'ber' in metrics_to_compute:
                    metrics['ber'].update(res, gt)
                if 'acc' in metrics_to_compute:
                    metrics['acc'].update(res, gt)
                if 'hce' in metrics_to_compute:
                    metrics['hce'].step(pred=res, gt=gt, gt_ske=ske_ary)

            # Show results for selected metrics
            if 'mae' in metrics_to_compute:
                results['mae'] = metrics['mae'].show()
            if 'fm' in metrics_to_compute:
                maxf, meanf, _, _ = metrics['fm'].show()
                results['maxf'] = maxf
                results['meanf'] = meanf
            if 'sm' in metrics_to_compute:
                results['sm'] = metrics['sm'].show()
            if 'em' in metrics_to_compute:
                results['em'] = metrics['em'].show()
            if 'wfm' in metrics_to_compute:
                results['wfm'] = metrics['wfm'].show()
            if 'm_dice' in metrics_to_compute:
                results['m_dice'] = metrics['m_dice'].show()
            if 'm_iou' in metrics_to_compute:
                results['m_iou'] = metrics['m_iou'].show()
            if 'ber' in metrics_to_compute:
                results['ber'] = metrics['ber'].show()
            if 'acc' in metrics_to_compute:
                results['acc'] = metrics['acc'].show()
            if 'hce' in metrics_to_compute:
                results['hce'] = metrics['hce'].get_results()["hce"]

            print(
                'dataset: {} MAE: {:.4f} Ber: {:.4f} maxF: {:.4f} avgF: {:.4f} wfm: {:.4f} Sm: {:.4f} adpEm: {:.4f} M_dice: {:.4f} M_iou: {:.4f} Acc: {:.4f} HCE:{}'.format(
                    name, results.get('mae', 0), results.get('ber', 0), results.get('maxf', 0),
                    results.get('meanf', 0), results.get('wfm', 0), results.get('sm', 0), results.get('em', 0),
                    results.get('m_dice', 0), results.get('m_iou', 0), results.get('acc', 0),
                    int(results.get('hce', 0)))
            )

    return results

if __name__ == '__main__':
    # ['mae', 'fm', 'sm', 'em', 'wfm', 'm_dice', 'm_iou', 'ber', 'acc', 'hce']
    metrics_to_compute = ['mae', 'sm', 'em',  'hce', 'm_iou','fm','wfm']
    results = evaluate_metrics(pred, splits, metrics_to_compute)
    print('MAE', results['mae'])
    print('sm', results['sm'])
    print('em', results['em'])
    print('hce', results['hce'])
    print('m_iou', results['m_iou'])
    print('fm', results['fm'])
    print('wfm', results['wfm'])

