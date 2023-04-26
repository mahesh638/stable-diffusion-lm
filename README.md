# stable-diffusion-lm
 
1) Download the `cc-news-train.txt` from the releases section into `cc_news/cc_news`

2) Download the required packages by running the code: `pip install -r requirements.txt`

3) Run the model by running the command: `bash scripts/train.sh`

4) After the model is run, a checkpoint folder with the model parameters are created to generate text. Run this command to generate text: `bash scripts/text_sample.sh ckpts/cc_news/ema_0.9999_025000.pt 2000 10`. Change the checkpoint to the generated checkpoint that you wish to visualize results for.
