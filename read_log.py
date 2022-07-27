import re
import numpy as np


def read_log(log_dir,print_out=True):

    with open(log_dir+'/log.txt',"r") as f:
        ff = f.read()

    his = ff.split("Loss")
    his.pop(0)

    dicts = [{"mean_accuracy":[],"worst_accuracy":[]} for i in range(3)]
    train_dict,test_dict,val_dict = dicts
    for i in his:
        tvt = i.split('results')
        tvt.pop(0)
        
        for k in range(3):
            content = tvt[k]
            pos_mean = re.search('mean_accuracy', content).span()
            mean_acc = content[pos_mean[1]+3:pos_mean[1]+21]
            pos_worst = re.search('worst_accuracy', content).span()
            worst_acc = content[pos_worst[1]+3:pos_worst[1]+21]
            dicts[k]["mean_accuracy"].append(mean_acc)
            dicts[k]["worst_accuracy"].append(worst_acc)

    mean_val_acc = np.array(val_dict['mean_accuracy'])
    worst_val_acc = np.array(val_dict['worst_accuracy'])
    mean_test_acc = np.array(test_dict['mean_accuracy'])
    worst_test_acc = np.array(test_dict['worst_accuracy'])
    i = np.argmax(mean_val_acc)
    k = np.argmax(worst_val_acc)
    if print_out:
        print("Experiment:",log_dir)
        print('Use best val mean acc:\n',f"Test Worst Acc: {worst_test_acc[i]}, Test Mean Acc: {mean_test_acc[i]}")
        print('Use best val worst acc:\n',f"Test Worst Acc: {worst_test_acc[k]}, Test Mean Acc: {mean_test_acc[k]}")
        print('Use last epoch:\n',f"Test Worst Acc: {worst_test_acc[-1]}, Test Mean Acc: {mean_test_acc[-1]}")

    return {'best_val_mean':[worst_test_acc[i],mean_test_acc[i]],'best_val_worst':[worst_test_acc[k],mean_test_acc[k]],'last':[worst_test_acc[-1],mean_test_acc[-1]]}

def get_average(res_lis):
    n_avr = len(res_lis)
    res_avr = {'best_val_mean':[0,0],'best_val_worst':[0,0],'last':[0,0]}
    var = {'best_val_mean':[[0,1],[0,1]],'best_val_worst':[[0,1],[0,1]],'last':[[0,1],[0,1]]}
    vrange = {'best_val_mean':[0,0],'best_val_worst':[0,0],'last':[0,0]}
    for i in res_lis:
        for save_type in i.keys():
            for n in range(2):
                # print(save_type,float(re.sub(u"([^\u0030-\u0039\u002e\uffe5])", "", i[save_type][n])))
                curr_acc = float(re.sub(u"([^\u0030-\u0039\u002e\uffe5])", "", i[save_type][n]))
                res_avr[save_type][n] += curr_acc
                var[save_type][n][0] = max(var[save_type][n][0],curr_acc)
                var[save_type][n][1] = min(var[save_type][n][1],curr_acc)
    for save_type in res_avr.keys():
        for n in range(2):
            res_avr[save_type][n] = res_avr[save_type][n] / n_avr
    # for save_type in res_avr.keys():
    #     for n in range(2):
    #         vrange[save_type][n] = var[save_type][n][0] - 

    print("average result:")
    print(res_avr)
    # print("result variance:")
    # print(var)
    # print("range:")
    # print(vrange)
    return res_avr, var

if  __name__ == '__main__':
    log_dir = '/datasets/jianhaoy/experimentwk0/celebA/styleclipallf3upsample5_42_sch'
    # log_dir_lis = ['/datasets/jianhaoy/experimentwk0/waterbird/glidef3_40_sch','/datasets/jianhaoy/experimentwk0/waterbird/glidef3_41_sch','/datasets/jianhaoy/experimentwk0/waterbird/glidef3_42_sch']
    # log_dir_lis = ['/datasets/jianhaoy/experimentwk0/waterbird/baseline41_sch','/datasets/jianhaoy/experimentwk0/waterbird/baseline42_sch','/datasets/jianhaoy/experimentwk0/waterbird/baseline43_sch']
    # log_dir_lis = ['/datasets/jianhaoy/experimentwk0/waterbird/vqganf3_43_sch','/datasets/jianhaoy/experimentwk0/waterbird/vqganf3_42_sch','/datasets/jianhaoy/experimentwk0/waterbird/vqganf3_41_sch']
    # log_dir_lis = ['/datasets/jianhaoy/experimentwk0/celebA/baseline43_sch','/datasets/jianhaoy/experimentwk0/celebA/baseline42_sch','/datasets/jianhaoy/experimentwk0/celebA/baseline41_sch']
    # log_dir_lis = ['/datasets/jianhaoy/experimentwk0/celebA/styleclipallf4_43_sch','/datasets/jianhaoy/experimentwk0/celebA/styleclipallf4_42_sch','/datasets/jianhaoy/experimentwk0/celebA/styleclipallf4_41_sch']
    # log_dir_lis = ['/datasets/jianhaoy/experimentwk0/celebA/styleclipallf4_43_sch','/datasets/jianhaoy/experimentwk0/celebA/styleclipallf4_41_sch','/datasets/jianhaoy/experimentwk0/celebA/styleclipallf4_42_sch']
    see_single = True
    if see_single:
        #Read single log
        _ = read_log(log_dir,print_out=True)
    else:
        #Read multiple and calculate average
        res_lis = []
        print(log_dir_lis)
        for ldr in log_dir_lis:
            res = read_log(ldr,print_out=False)
            res_lis.append(res)
        # print(res_lis)
        res_avr,var = get_average(res_lis)
