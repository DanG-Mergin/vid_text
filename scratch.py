# import re

# with open("1_PCA.txt", 'r') as file:
#     i = str(file.read())

# regex = re.compile(r'[^a-zA-Z\s]')
# i = regex.sub('', i)
# i = i.replace('\n', '')

# with open("1_PCA2.txt", "w") as f:
#     f.write(i)
# import coqui_stt_training
# from coqui_stt_training.train import train

# train.auto_input_dataset('coqui_train_PCA_1_0_1380')