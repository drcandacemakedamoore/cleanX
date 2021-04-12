# cleanX
Python library for cleaning large datasets of Xrays.

primary author: Dr. Candace Makeda H. Moore, MD
other authors: Oleg Sivokon, Andrew Murphy

Includes several functions including: 


Ones to run on dataframes to make sure there is no image leakage: 

check_paths_for_group_leakage(train_df, test_df, uniqueID):

    """
    Args:
        train_df (dataframe): dataframe describing train dataset
        test_df (dataframe): dataframe describing test dataset
        uniqueID (str): string name of column with image ID, patient IDs or some other unique ID that is in all dfs
    
    Returns:
        pics_in_both_groups: duplications of any image into both sets as a new dataframe
    """
    
    
One to run on single images, one at a time, if you want to crop off a black frame:

dcrop(image):

     """
    Args:
        
        image: an image 
    
    Returns:
        image[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]: image cropped of black edges
    """
    
   
One to run on a list to make a prototype tiny Xray others can be comapared to: 


seperate_image_averger(set_of_images, s=5 ):

    """
    Args:
        
        set_of_images: a list 
        s: number of pixels for height and wifth
    
    Returns:
        canvas/len(set_of_images): an average tiny image (can feed another function which compares to this mini)
    """
    
Many to run on image files which are inside a folder to check if they are "clean"

augment_and_move(origin_folder, target_folder, transformations):
    
    """
    Args:
        origin_folder: folder with 'virgin' images
        target_folder: folder to drop images after transformations
        transformations : example tranformations = [ImageOps.mirror, ImageOps.flip]...some function to transform the image
    
    Returns:
        pics_in_both_groups: duplications of any image into both sets as a new dataframe
    """
   


find_by_sample_upper(source_directory, percent_height_of_sample,  value_for_line):
 
    """

    function that takes top (upper percent) of images and checks if average pixel value is above value_for_line
        """         

find_sample_upper_greater_than_lower(source_directory, percent_height_of_sample):
 
    """
    function that checks that upper field (cut on percent_height of sample) of imagae has a higher pixel value than the lower field (it should in a typical CXR)
     
    """
    
def find_outliers_by_total_mean(source_directory, percentage_to_say_outliers):

        """
        Args:
        source_directory: directory with image files (should be more than 20)
        percentage_to_say_outliers: a number which will be the percentage of images contained in 
        the high mean and low mean sets
    
        Returns:
        lows,highs: images with low mean, images with high mean
        """
        


find_outliers_by_mean_to_df(source_directory, percentage_to_say_outliers):

        """
        Important note: approximate, and it can by chance cut the group so images with 
        the same mean are in and out of normal range if the knife so falls
        
        Args:
        source_directory: directory with image files (should be more than 20)
        percentage_to_say_outliers: a number which will be the percentage of images contained in 
        the high mean OR low mean sets- note if you set to 50, then all images will be high or low
    
        Returns:
        lows,highs: images with low mean, images with high mean into a dataframe
        """
        


find_tiny_image_differences(directory, s=5, percentile=8): 

    """
    Note: percentile returned is approximate, may be a tad more 
    Args:
        directory: directory of all the images you want to compare
        s: size of image sizes to compare
        percentile: what percentage you want to return
    Returns:
        difference_outliers: outliers in terms of difference from an average image
    """
      

tesseract_specific(directory):

 
    """this function runs tessseract ocr for text detection over images in a directory, and gives a dataframe with what it found"""
   

find_suspect_text(directory, label_word):
 
    """finds a specific string you believe is a label e.g. "cancer"  , this function looks for one single string in texts (multilingual!) on images

     
    """

find_suspect_text_by_legnth(directory, legnth):
 
    """
     this function finds all texts above a specified legnth (number of charecters)
      
    """
   
histogram_difference_for_inverts(directory):
 
    """
     this function looks for images by a spike on the end of pixel value histogram to find inverted images
      
    """
          
histogram_difference_for_inverts_todf(directory):
    

find_duplicated_images(directory):
 
    """
     this function finds duplicated images and return a list
      
    """
   
find_duplicated_images_todf(directory):
 
    """
     looks for duplicated images, returns dataframe
     
    """

Function that takes a dataframe and returns plotted images:

show_images_in_df(iter_ob, legnth_name):

    """
    Args:
        iter_ob: should be list(df.column)
        legnth_name: size of image name going from end
    Returns: plot of images with names    
        """
    
           
