---
license: cc-by-4.0
---

## Unitree G1 Fruits Pick and Place 1K Dataset

## Dataset Description:
The PhysicalAI-Robotics-GR00T-Teleop-G1 dataset consists of1000 teleoperation trajectories of real robot data using Unitree G1, with upper body control. The robot chooses the correct fruit to pick and place on the plate according to the language prompt. A total of 4 fruits are used: Apple, Pear, Starfruit, Grape. The robot is equipped with the default realsense camera, and a pair of Unitree G1 Tri-fingers hand.


<img src="https://cdn-uploads.huggingface.co/production/uploads/67b8da81d01134f89899b4a7/nnQttu2PywkTLVMcuNL42.jpeg" width="48%">


This dataset is ready for commercial/non-commercial use.

## Dataset Owner(s):
NVIDIA GEAR

## Dataset Creation Date:
June 1, 2025

## License/Terms of Use: 
This dataset is licensed under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/legalcode).

## Intended Usage:
Researchers, Academics, Open-Source Community: AI-driven robotics research and algorithm development.
Developers: Integrate and customize AI for various robotic applications.
Startups & Companies: Accelerate robotics development and reduce training costs.

## Dataset Characterization
**Data Collection Method**<br>
[Human] <br>

**Labeling Method**<br>
[Human] <br>

## Dataset Format
MP4 and HDF5 files

Modality:
 - Observation: 43 dim of vectorized state (joint positions of full body + hands)
 - Action: 43 dim of vectorized action (joint positions of full body + hands)
 - Video: RGB video, 640x480 resolution, 20fps
 - Language Instruction: 
   - *"Pick the apple from the table and place it into the basket."*
   - *"Pick the pear from the table and place it into the basket."*
   - *"Pick the grapes from the table and place them into the basket."*
   - *"Pick the starfruit from the table and place it into the basket."*


| Dataset Folder | Preview |
|----------------|---------|
| g1-pick-apple | ![image/png](https://cdn-uploads.huggingface.co/production/uploads/67b8da81d01134f89899b4a7/y8-gdvfIRhKtxbcQCYYRv.png) |
| g1-pick-pear | ![image/png](https://cdn-uploads.huggingface.co/production/uploads/67b8da81d01134f89899b4a7/zYqxhAGHV26raIAh9p1BS.png) |
| g1-pick-grapes | ![image/png](https://cdn-uploads.huggingface.co/production/uploads/67b8da81d01134f89899b4a7/a2ePP_DycQl7cjoV86QR6.png) |
| g1-pick-starfruit | ![image/png](https://cdn-uploads.huggingface.co/production/uploads/67b8da81d01134f89899b4a7/2260l3TdVqCM3R9FOCAMo.png) |

## Dataset Quantification
- 1000 teleoperation trajectories<br>
- 1000 videos in MP4<br>
- 1000 HDF5 files for actions<br>
- Total data storage: ~400MB<br>

## Download the dataset

```bash
huggingface-cli download \
    --repo-type dataset nvidia/PhysicalAI-Robotics-GR00T-Teleop-G1 \
    --local-dir ./datasets/
```

## Finetuning with GR00T-N1.5

Refer to the [Github repo](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/3_0_new_embodiment_finetuning.md) for the finetuning script.

## Ethical Considerations:
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications.  When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.   

Please report security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).