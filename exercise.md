The guiding video for this exercise is the rest of Karpathy’s [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU) (from minute 46 onwards). However, we will not be implementing everything in the same order he does in the video, and there are readings attached to supplement the video.

**Workshop Goal:** Get the smoke-test passing. Then progress in the exercise as much as you can

**Exercise Goal:** Run an optimized training loop on a single L40S GPU, with mixed-precision training and flash-attention (in the next exercise we will run distributed training on 8 H100 GPUs)

## 1.2.0 Setup on Nebius (During class)

**total time: 20-40 min**

Fork the repo. Keep the repo public so we can download it to the Nebius job

Download the private key that will be sent at the lesson.

First [install Nebius CLI](https://docs.nebius.com/cli/install) 

Now set up authentication to nebius (replace \<your-name\> and \<path-to-private.pem\>)

nebius profile create \\  
                                                        \--endpoint api.nebius.cloud \\  
                                                        \--service-account-id serviceaccount-e00j9yyxp97astchac \\  
                                                        \--public-key-id publickey-e00cacpw3qngckqsqm \\  
                                                        \--private-key-file \<path-to-private.pem\> \\  
                                                        \--profile \<your-name\>\\  
                                                        \--parent-id project-e00qcp0kpr00gb7cnvmav8

Run the smoke test on a single L40S GPU

nebius ai job create \\  
                                        \--name smoke-test-\<your-name\> \\  
                                        \--image cr.eu-north1.nebius.cloud/e00v1er5fasm8gmdwy/apex-ex-1 \\  
                                        \--container-command bash \\  
                                        \--args '-c "git clone https://github.com/Apex-IL/architects-ex-1.git \-b smoke-test && cd architects-ex-1 && python train\_gpt2.py"' \\  
                                        \--platform gpu-l40s-a \\  
                                        \--preset 1gpu-16vcpu-64gb \\  
                                        \--timeout 15m \\  
                                        \--volume computefilesystem-e00hnnpfn5rr5aavma:/mnt/data

Please use `--timeout` in all your jobs, and avoid timeouts longer than 30m to preserve resources.

## 1.2.1 Basic training loop

**total time:** **1-2 hr**

**Relevant parts in video: 00:46:00-1:22:00**

Write a training loop to be run on a single L40S GPU.

Fork the exercise repository at [https://github.com/Apex-IL/architects-ex-1/tree/master](https://github.com/Apex-IL/architects-ex-1/tree/master) . Then you can change the code in train\_gpt2.py

We recommend setting B=4 and T=64 so you don’t run out of memory at this stage. Later we will increase that

Since training jobs take time to start on nebius, you may find it more comfortable to run on colab at this stage. Note you will need to download the fineweb dataset (which on nebius we attach as a volume) which takes significant time.

Command to run (replace \<your-name\> and \<your-repo-url.git\>):

nebius ai job create \\  
                                        \--name ex1-\<your-name\> \\  
                                        \--image cr.eu-north1.nebius.cloud/e00v1er5fasm8gmdwy/apex-ex-1 \\  
                                        \--container-command bash \\  
                                        \--args '-c "git clone \<your-repo-url.git\> && cd architects-ex-1 && python train\_gpt2.py"' \\  
                                        \--platform gpu-l40s-a \\  
                                        \--preset 1gpu-16vcpu-64gb \\  
                                        \--timeout 15m \\  
                                        \--volume computefilesystem-e00hnnpfn5rr5aavma:/mnt/data

## 1.2.2 mixed precision training

**total time: 45 min**

**Relevant part in video: 1:22:00-1:49:00**

This is a small addition that allows us to utilize the compute better.

## 1.2.3 torch.compile

**total time: 20 min**

**Relevant part in video: 1:49:00-2:00:00**

This is another small change that massively impacts performance

## 1.2.4 Flash Attention

**total time: 20 min**  
**Relevant part in video: 2:00:00-2:08:00**

Replace your attention implementation with pytorch built-in flash attention implementation.

