import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline

def SummarizeResult(result_type: str = 'image', file_paths: list = ['Not_Specified'],
                    subplot_titles: list = ['Not_Specified'], figtitle: str = 'Not_Specified', n_column: int = 2):
    if result_type == 'image':
        if file_paths == ['Not_Specified']:
            print('Please specify the paths to read in the results first.')
        else:
            images = []
            for file_path in file_paths:
                images.append(mpimg.imread(file_path))
            columns = n_column
            fig,ax = plt.subplots(len(images)//columns,columns,figsize=(10*columns,7*(len(images)//columns)))
            fig.tight_layout()
            for i, image in enumerate(images):
                ax[i//2,i-(i//2)*2] = plt.subplot(len(images) / columns + 1, columns, i + 1)
                ax[i//2,i-(i//2)*2].axis('off')
                ax[i//2,i-(i//2)*2].set_title(subplot_titles[i])
                plt.imshow(image)
            if len(images)%2 != 0:
                n_blank = columns - (len(images)%2)
                ax[-1, -(n_blank):].axis('off')
            fig.suptitle(figtitle, y=1.03, fontsize=18, verticalalignment='top')
