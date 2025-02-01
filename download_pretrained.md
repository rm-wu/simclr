# Pretrained Models

### CriBo

<table class="center">
  <tr>
    <th>pretraining dataset</th>
    <th>arch</th>
    <th>params</th>
    <th>batchsize</th>
    <th>Dense NN retrieval ADE20k (mIoU)</th>
    <th>Dense NN retrieval PVOC12 (mIoU)</th>
    <th colspan="2">download</th>
  </tr>

  <tr>
    <th>COCO</th>
    <th>ViT-S/16</th>
    <th>21M</th>
    <th>256</th>
    <th>23.4</th>
    <th>58.1</th>
    <td><a href="https://rdr.kuleuven.be/api/access/datafile/159667">ckpt</a></td>
    <td><a href="https://github.com/tileb1/CrIBo/blob/main/checkpoints/vits16-coco_args.json">args</a></td>
  </tr>

  <tr>
    <th>ImageNet-1k</th>
    <th>ViT-S/16</th>
    <th>21M</th>
    <th>1024</th>
    <th>28.3</th>
    <th>73.2</th>
    <td><a href="https://rdr.kuleuven.be/api/access/datafile/159665">ckpt</a></td>
    <td><a href="https://github.com/tileb1/CrIBo/blob/main/checkpoints/vits16-in_args.json">args</a></td>
  </tr>
  <tr>
    <th>ImageNet-1k</th>
    <th>ViT-B/16</th>
    <th>85M</th>
    <th>1024</th>
    <th>30.0</th>
    <th>74.9</th>
    <td><a href="https://rdr.kuleuven.be/api/access/datafile/159666">ckpt</a></td>
    <td><a href="https://github.com/tileb1/CrIBo/blob/main/checkpoints/vitb16-in_args.json">args</a></td>
  </tr>
</table>


``` bash
mkdir checkpoints


wget -O checkpoints/cribo_vits16_coco.pth https://rdr.kuleuven.be/api/access/datafile/159667
wget -O checkpoints/cribo_vits16_in1k.pth https://rdr.kuleuven.be/api/access/datafile/159665
wget -O checkpoints/cribo_vitb16_in1k.pth https://rdr.kuleuven.be/api/access/datafile/159666
````
