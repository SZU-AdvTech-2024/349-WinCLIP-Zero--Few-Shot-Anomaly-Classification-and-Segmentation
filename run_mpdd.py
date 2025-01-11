import os
from multiprocessing import Pool

if __name__ == '__main__':
    pool = Pool(processes=1)  # 进程池

    sh_method = f'python main.py ' \
                f'--dataset mpdd ' \
                f'--data_path /mnt/d/anomaly/mpdd ' \
                f'--save_path ./results/winclip_mpdd_0_shots ' \
                f'--model ViT-B-16-plus-240 ' \
                f'--pretrained openai ' \
                f'--k_shot 0 ' \
                f'--image_size 240 ' \

    print(sh_method)
    pool.apply_async(os.system, (sh_method,))
    sh_method = f'python main.py ' \
                f'--dataset mpdd ' \
                f'--data_path /mnt/d/anomaly/mpdd ' \
                f'--save_path ./results/winclip_mpdd_1_shots ' \
                f'--model ViT-B-16-plus-240 ' \
                f'--pretrained openai ' \
                f'--k_shot 1 ' \
                f'--image_size 240 ' \

    print(sh_method)
    pool.apply_async(os.system, (sh_method,))
    sh_method = f'python main.py ' \
                f'--dataset mpdd ' \
                f'--data_path /mnt/d/anomaly/mpdd ' \
                f'--save_path ./results/winclip_mpdd_2_shots ' \
                f'--model ViT-B-16-plus-240 ' \
                f'--pretrained openai ' \
                f'--k_shot 2 ' \
                f'--image_size 240 ' \

    print(sh_method)
    pool.apply_async(os.system, (sh_method,))
    sh_method = f'python main.py ' \
                f'--dataset mpdd ' \
                f'--data_path /mnt/d/anomaly/mpdd ' \
                f'--save_path ./results/winclip_mpdd_4_shots ' \
                f'--model ViT-B-16-plus-240 ' \
                f'--pretrained openai ' \
                f'--k_shot 4 ' \
                f'--image_size 240 ' \

    print(sh_method)
    pool.apply_async(os.system, (sh_method,))
    pool.close()
    pool.join()  # 等待进程结束
 