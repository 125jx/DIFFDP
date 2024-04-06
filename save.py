import SimpleITK as sit
import numpy as np
import os
# from record import comparison
def save_one(pre,real,path,name):
    pre[real==0]=0
    fakes: sit.Image = sit.GetImageFromArray(pre)
    fakes.SetSpacing(spacing=(3, 3, 3))
    s = sit.ImageFileWriter()
    s.SetFileName(path + '\\' + name+ '_pre.mha')
    s.Execute(fakes)
    sit.WriteImage(fakes, path + '\\' +name+ '_pre.mha')

    reals: sit.Image = sit.GetImageFromArray(real)
    reals.SetSpacing(spacing=(3, 3, 3))
    s = sit.ImageFileWriter()
    s.SetFileName(path + '\\' + name+ '_tar.mha')
    s.Execute(reals)
    sit.WriteImage(reals, path + '\\' + name+ '_tar.mha')

    diffs: sit.Image = sit.GetImageFromArray(np.clip(abs(real-pre)*3,0.,1.))
    diffs.SetSpacing(spacing=(3, 3, 3))
    s = sit.ImageFileWriter()
    s.SetFileName(path + '\\' + name + '_diff.mha')
    s.Execute(diffs)
    sit.WriteImage(diffs, path + '\\' + name + '_diff.mha')

def two2one(pre1,pre2,real,path,name):
    fake = np.zeros_like(real)
    for i in range(len(pre1)):
        p1 = pre1[i]
        p2 = pre2[i]
        gt = real[i]
        # print(abs(p1-gt).mean(),abs(p2-gt).mean())
        if abs(p1-gt).mean() >= abs(p2-gt).mean() and abs(p1-gt).mean()<=abs(p2-gt).mean()+0.008 :
            # print(11111)
            fake[i] = p1
        else:
            fake[i] = p2

    save_one(fake,real,path,name)

def diffmap(pre,real,path,name):
    diffs: sit.Image = sit.GetImageFromArray(np.clip(abs(real - pre) * 3, 0., 1.))
    diffs.SetSpacing(spacing=(3, 3, 3))
    s = sit.ImageFileWriter()
    s.SetFileName(path + '\\' + name + '_diff.mha')
    s.Execute(diffs)
    sit.WriteImage(diffs, path + '\\' + name + '_diff.mha')


if __name__ == '__main__':
    # path = r"ablation\mha\base_enc7"
    # path1 = r"ablation\mha\base_enc4"
    # path2 = r"ablation\mha\base_enc5"
    # if not os.path.exists(path):
    #     os.makedirs(path)
    # for root,_,files in os.walk(path1):
    #     for f in files:
    #         if f.find('pre')>0:
    #             pre1 = sit.GetArrayFromImage(sit.ReadImage(os.path.join(path1,f)))
    #             pre2 = sit.GetArrayFromImage(sit.ReadImage(os.path.join(path2, f)))
    #
    #             real = sit.GetArrayFromImage(sit.ReadImage(os.path.join(path1, f.replace('pre', 'tar'))))
    #             name = f.split('_')[0]
    #             print(name)
    #             two2one(pre1,pre2,real,path,name)


    path = r"comparison/mha"
    path1 = os.path.join(path,comparison['mcgan']['mha'])


    for root, _, files in os.walk(path1):
        for f in files:
            if f.find('pre') > 0:
                pre1 = sit.GetArrayFromImage(sit.ReadImage(os.path.join(path1, f)))


                real = sit.GetArrayFromImage(sit.ReadImage(os.path.join(path1, f.replace('pre', 'tar'))))
                name = f.split('_')[0]
                print(name)
                diffmap(pre1 ,real, path1, name)