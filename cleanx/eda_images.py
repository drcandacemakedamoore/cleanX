
def avg_image_maker(set_of_images):
    """
    :param set_of_images: A set of images, can be read in with glob.glob on a folder of jpgs
    :type set_of_images: list    
    
    :return: final_avg, an image that is the average image of images in the set
    :rtype: nd.array
    """
    list_h = []
    list_w = []
    
    for example in set_of_images:
        example = cv2.imread(example, cv2.IMREAD_GRAYSCALE)
        ht= image.shape[0]
        wt= image.shape[1]
        list_h.append(ht)  
        list_w.append(wt)
        
    h = int(sum(list_h)/len(list_h))    
    w = int(sum(list_w)/len(list_w))
    canvas = np.zeros((h, w))
    for example in set_of_images:
        example = cv2.imread(example, cv2.IMREAD_GRAYSCALE)    
        example_small = cv2.resize(example, (w, h))
        canvas += np.array(example_small)
    final_avg = canvas / len(set_of_images)    
    return final_avg