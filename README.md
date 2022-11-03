# Fine Tuning Stable Diffusion

> Hey, guys! This is my second project on Chinese Landscape Painting Generation.  Hopefully you can enjoy it!

**Chinese Landscape Painting** is a kind of Chinese traditional art form which is totally different from western art. However, just like western art, it have both aesthetic and artistic value. For me, a Chinese student who is just getting started with AI, it sounds great to combine edge-cutting technique with traditional culture. 

I follow [the tutorial of LambdaLabsML]([examples/stable-diffusion-finetuning at main · LambdaLabsML/examples (github.com)](https://github.com/LambdaLabsML/examples/tree/main/stable-diffusion-finetuning)), they generated excellent Pokémon. What if I use the same method in Chinese Landscape painting? Here are some examples of the sort of outputs the trained model can produce, and the prompt used: 

> They look nice, model has learned enough features of Landscape Paintings obviously!

If you're just after the model, code, or dataset, see:

- [stable-diffusion](https://github.com/CompVis/stable-diffusion)

- [Lambda Diffusers](https://github.com/LambdaLabsML/lambda-diffusers)
- [Training code](https://github.com/justinpinkney/stable-diffusion)

## Hardware

Just as the tutorial said, running Stable Diffusion itself is not too demanding by today's standards, and fine tuning the model doesn't require anything like the hardware on which it was originally trained. 

Here, I use 1xA6000 GPUs on [恒源云](https://gpushare.com/store/hire?create=true) (a Chinese GPU sharing platform) and run training for around 35 steps which takes about nearly 12 hours to run.  Training should be able to run on a single or lower spec GPUs (as long as there is >24GB of VRAM), but you might need to adjust batch size and gradient accumulation steps to fit your GPU. For more details on training code please see the [fine-tuning notebook](https://github.com/LambdaLabsML/examples/blob/main/stable-diffusion-finetuning/pokemon_finetune.ipynb).

## Data!

First off we need a dataset to train on. I cramped 3160 images from searcher engines (i.e. baidu and bing), and got the captions by [BLIP](https://github.com/salesforce/BLIP) (actually I used [LAVIS](https://github.com/salesforce/LAVIS) to load it.). The captions aren't perfect, but they're reasonably accurate and good enough for our purposes. Btw, to emphasize **Chinese landscape painting**, I have replaced "painting" or "drawing" with a "Chinese landscape painting" in each text. 

```python
from datasets import load_dataset

sample = load_dataset("imagefolder", data_dir="home/to/your/path", split="train")
display(sample["image"].resize((256, 256)))
print(sample["text"])
```

> 'a chinese lanscape painting of a waterfall and trees'

## Get ready!

Now we have a dataset we need the original model weights which are available for [download here](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original), listed as `sd-v1-4-full-ema.ckpt`. Next we need to set up the code and environment for training. We're going to use a fork of the original training code which has been modified to make it a bit more friendly for fine-tuning purposes: [justinpinkney/stable-diffusion](https://github.com/justinpinkney/stable-diffusion).

Stable Diffusion uses yaml based configuration files along with a few extra command line arguments passed to the `main.py` function in order to launch training.

We've created a [base yaml configuration file](https://github.com/justinpinkney/stable-diffusion/blob/main/configs/stable-diffusion/pokemon.yaml) that runs this fine-tuning example. If you want to run on your own dataset it should be simple to modify, the main part you would need to edit is the data configuration, here's the relevant excerpt from the custom yaml file:

```
data:
  target: main.DataModuleFromConfig
  params:
    batch_size: 4
    num_workers: 4
    num_val_workers: 0 # Avoid a weird val dataloader issue
    train:
      target: ldm.data.simple.hf_dataset
      params:
        name: Chinese-Landscape-Painting-Style-Transfer
        image_transforms:
        - target: torchvision.transforms.Resize
          params:
            size: 512
            interpolation: 3
        - target: torchvision.transforms.RandomCrop
          params:
            size: 512
        - target: torchvision.transforms.RandomHorizontalFlip
    validation:
      target: ldm.data.simple.TextOnly
      params:
        captions:
        - "a chinese lanscape painting of a landscape with mountains in the background"
        - "a chinese lanscape painting of a building with trees in front of it"
        - "a chinese lanscape painting of a mountain landscape with trees"
        - "a chinese lanscape painting of a mountain scene with trees and a bridge"
        output_size: 512
        n_gpus: 1 # small hack to sure we see all our samples
```

## Train!

```
# Run training
!(python main.py \
    -t \
    --base configs/stable-diffusion/landscape_paintings.yaml \
    --gpus 0 \
    --scale_lr False \
    --num_nodes 1 \
    --check_val_every_n_epoch 1 \
    --finetune_from ckpt_path \
    data.params.batch_size=BATCH_SIZE \
    lightning.trainer.accumulate_grad_batches=ACCUMULATE_BATCHES \
    data.params.validation.params.n_gpus=N_GPUS \
)
```

## Test!

```
# Run the model
!(python scripts/txt2img.py \
    --prompt 'a chinese landscape painting of a landscape with mountains and a river' \
    --outdir 'outputs/generated_pl' \
    --H 512 --W 512 \
    --n_samples 4 \
    --config 'configs/stable-diffusion/landscape_paintings.yaml' \
    --ckpt 'logs/2022-11-01T17-08-20_landscape_paintings/checkpoints/last.ckpt')
```

```
from PIL import Image
im = Image.open("outputs/generated_pl/grid-0000.png").resize((1024, 256))
display(im)
print("a chinese landscape painting of a landscape with mountains and a river")
```

## Conclusion

OOH! That's interesting!