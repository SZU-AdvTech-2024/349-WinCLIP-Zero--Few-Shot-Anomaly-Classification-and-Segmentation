import os
from multiprocessing import Pool

if __name__ == '__main__':
    pool = Pool(processes=1)  # 进程池

    # sh_method = f'python main.py ' \
    #             f'--dataset mvtec ' \
    #             f'--data_path /mnt/d/anomaly/mvtec_anomaly_detection ' \
    #             f'--save_path ./results/winclip_mvtec_0_shots ' \
    #             f'--model ViT-B-16-plus-240 ' \
    #             f'--pretrained openai ' \
    #             f'--k_shot 0 ' \
    #             f'--image_size 240 ' \
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
    pool.close()
    pool.join()  # 等待进程结束

