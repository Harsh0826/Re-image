from django.shortcuts import render
import requests

# 3rd video data
from django.core.files.storage import FileSystemStorage

# Function will be called when first time website loads
def button(request):
    return render(request, 'index.html')

    # ----------------------------------------------------------------------

import sys
from subprocess import run,PIPE

def index_method(request):
    myrange = request.POST.get('myRange')
    mywidth = request.POST.get('width')
    myheight = request.POST.get('height')

    print("This are all VALUES : ",myrange)
    print("This are all VALUES : ",mywidth)
    print("This are all VALUES : ",myheight)

    
    # 3rd video data
    image = request.FILES['image']
    print("Image is ", image)
    fs = FileSystemStorage()
    filename = fs.save(image.name,image)
    fileurl = fs.open(filename)
    templateurl = fs.url(filename)
    print("file raw url",filename)
    print("file full url", fileurl)
    print("template url",templateurl)

    fileurl = str(fileurl)
    filename = str(filename)
    # image_format = str(image_format)

    from PIL import Image
    im = Image.open(fileurl).convert("RGB")

    image_fullpath = fileurl
    image_name = filename
    print("Image FULL PATH : ", image_fullpath)
    print("Image FILE NAME : ", image_name)

    if mywidth == "0" and myheight == "0":
        range = int(myrange)/100
        resized_im = im.resize((round(im.size[0]*range), round(im.size[1]*range)))
        image_save_path = image_fullpath
        resized_im.save(image_save_path)
    else:
        width = float(mywidth)
        height = float(myheight)
        resized_im = im.resize((round(width), round(height)))
        image_save_path = image_fullpath
        resized_im.save(image_save_path)

    # **************************************************************

    image_save_path = "/media/"+image_name
    edit_image_url = str(image_save_path)

    return render(request,'index.html',
    {'raw_url':templateurl, 'edit_url':edit_image_url})






def image_compressor_method(request):
    image_quality = request.POST.get('image_quality')
    print("This is the image quality : ",image_quality)
    
    # 3rd video data
    image = request.FILES['image']
    print("Image is ", image)
    fs = FileSystemStorage()
    filename = fs.save(image.name,image)
    fileurl = fs.open(filename)
    templateurl = fs.url(filename)
    print("file raw url",filename)
    print("file full url", fileurl)
    print("template url",templateurl)

    fileurl = str(fileurl)
    filename = str(filename)
    # image_format = str(image_format)

    from PIL import Image
    im = Image.open(fileurl).convert("RGB")
    image_quality = int(image_quality)
    im.save(fileurl, optimize = True, quality = image_quality)

    # **************************************************************

    image_save_path = "/media/"+filename
    edit_image_url = str(image_save_path)
    # edit_image_url = edit_image_url[2:-5]

    return render(request,'image-compressor.html',
    {'raw_url':templateurl, 'edit_url':edit_image_url})







def online_image_converter_method(request):
    image_format = request.POST.get('image_format')
    print("This is the image format : ",image_format)
    
    # 3rd video data
    image = request.FILES['image']
    print("Image is ", image)
    fs = FileSystemStorage()
    filename = fs.save(image.name,image)
    fileurl = fs.open(filename)
    templateurl = fs.url(filename)
    print("file raw url",filename)
    print("file full url", fileurl)
    print("template url",templateurl)

    fileurl = str(fileurl)
    filename = str(filename)
    image_format = str(image_format)

    import sys
    from PIL import Image

    image_fullpath = fileurl
    image_name = filename
    image_format = image_format
    img = Image.open(str(image_fullpath)).convert("RGB")

    import string    
    import random # define the random module  
    S = 10  # number of characters in the string.  
    # call random.choices() string module to find the string in Uppercase + numeric data.  
    ran = ''.join(random.choices(string.ascii_uppercase + string.digits, k = S)) 

    image_save_path = image_fullpath.replace(image_name,str(ran)+"."+image_format)
    img.save(image_save_path,image_format)
    print("/media/temp."+image_format)

    # **************************************************************

    image_save_path = "/media/"+str(ran)+"."+image_format
    edit_image_url = str(image_save_path)

    return render(request,'online-image-converter.html',
    {'raw_url':templateurl, 'edit_url':edit_image_url})








def image_dehazer_method(request):
    
    # 3rd video data
    image = request.FILES['image']
    print("Image is ", image)
    fs = FileSystemStorage()
    filename = fs.save(image.name,image)
    fileurl = fs.open(filename)
    templateurl = fs.url(filename)
    print("file raw url",filename)
    print("file full url", fileurl)
    print("template url",templateurl)

    copy = templateurl

    fileurl = str(fileurl)
    filename = str(filename)
    # image_format = str(image_format)




    import cv2
    import copy
    import numpy as np


    #************************************************************************************

    def Airlight(HazeImg, AirlightMethod, windowSize):
        if(AirlightMethod.lower() == 'fast'):
            A = []
            if(len(HazeImg.shape) == 3):
                for ch in range(len(HazeImg.shape)):
                    kernel = np.ones((windowSize, windowSize), np.uint8)
                    minImg = cv2.erode(HazeImg[:, :, ch], kernel)
                    A.append(int(minImg.max()))
            else:
                kernel = np.ones((windowSize, windowSize), np.uint8)
                minImg = cv2.erode(HazeImg, kernel)
                A.append(int(minImg.max()))
        return(A)

    #****************************************************************************************8

    def BoundCon(HazeImg, A, C0, C1, windowSze):
        if(len(HazeImg.shape) == 3):

            t_b = np.maximum((A[0] - HazeImg[:, :, 0].astype(np.float)) / (A[0] - C0),
                            (HazeImg[:, :, 0].astype(np.float) - A[0]) / (C1 - A[0]))
            t_g = np.maximum((A[1] - HazeImg[:, :, 1].astype(np.float)) / (A[1] - C0),
                            (HazeImg[:, :, 1].astype(np.float) - A[1]) / (C1 - A[1]))
            t_r = np.maximum((A[2] - HazeImg[:, :, 2].astype(np.float)) / (A[2] - C0),
                            (HazeImg[:, :, 2].astype(np.float) - A[2]) / (C1 - A[2]))

            MaxVal = np.maximum(t_b, t_g, t_r)
            transmission = np.minimum(MaxVal, 1)
        else:
            transmission = np.maximum((A[0] - HazeImg.astype(np.float)) / (A[0] - C0),
                            (HazeImg.astype(np.float) - A[0]) / (C1 - A[0]))
            transmission = np.minimum(transmission, 1)

        kernel = np.ones((windowSze, windowSze), np.float)
        transmission = cv2.morphologyEx(transmission, cv2.MORPH_CLOSE, kernel=kernel)
        return(transmission)

    #************************************************************************************

    def CalTransmission(HazeImg, Transmission, regularize_lambda, sigma):
        rows, cols = Transmission.shape

        KirschFilters = LoadFilterBank()

        # Normalize the filters
        for idx, currentFilter in enumerate(KirschFilters):
            KirschFilters[idx] = KirschFilters[idx] / np.linalg.norm(currentFilter)

        # Calculate Weighting function --> [rows, cols. numFilters] --> One Weighting function for every filter
        WFun = []
        for idx, currentFilter in enumerate(KirschFilters):
            WFun.append(CalculateWeightingFunction(HazeImg, currentFilter, sigma))

        # Precompute the constants that are later needed in the optimization step
        tF = np.fft.fft2(Transmission)
        DS = 0

        for i in range(len(KirschFilters)):
            D = psf2otf(KirschFilters[i], (rows, cols))
            DS = DS + (abs(D) ** 2)

        # Cyclic loop for refining t and u --> Section III in the paper
        beta = 1                    # Start Beta value --> selected from the paper
        beta_max = 2**8             # Selected from the paper --> Section III --> "Scene Transmission Estimation"
        beta_rate = 2*np.sqrt(2)    # Selected from the paper

        while(beta < beta_max):
            gamma = regularize_lambda / beta

            # Fixing t first and solving for u
            DU = 0
            for i in range(len(KirschFilters)):
                dt = circularConvFilt(Transmission, KirschFilters[i])
                u = np.maximum((abs(dt) - (WFun[i] / (len(KirschFilters)*beta))), 0) * np.sign(dt)
                DU = DU + np.fft.fft2(circularConvFilt(u, cv2.flip(KirschFilters[i], -1)))

            # Fixing u and solving t --> Equation 26 in the paper
            # Note: In equation 26, the Numerator is the "DU" calculated in the above part of the code
            # In the equation 26, the Denominator is the DS which was computed as a constant in the above code

            Transmission = np.abs(np.fft.ifft2((gamma * tF + DU) / (gamma + DS)))
            beta = beta * beta_rate
        return(Transmission)

    def LoadFilterBank():
        KirschFilters = []
        KirschFilters.append(np.array([[-3, -3, -3],   [-3, 0, 5],   [-3, 5, 5]]))
        KirschFilters.append(np.array([[-3, -3, -3],   [-3, 0, -3],  [5, 5, 5]]))
        KirschFilters.append(np.array([[-3, -3, -3],   [5, 0, -3],   [5, 5, -3]]))
        KirschFilters.append(np.array([[5, -3, -3],    [5, 0, -3],   [5, -3, -3]]))
        KirschFilters.append(np.array([[5, 5, -3],     [5, 0, -3],   [-3, -3, -3]]))
        KirschFilters.append(np.array([[5, 5, 5],      [-3, 0, -3],  [-3, -3, -3]]))
        KirschFilters.append(np.array([[-3, 5, 5],     [-3, 0, 5],   [-3, -3, -3]]))
        KirschFilters.append(np.array([[-3, -3, 5],    [-3, 0, 5],   [-3, -3, 5]]))
        KirschFilters.append(np.array([[-1, -1, -1],   [-1, 8, -1],  [-1, -1, -1]]))
        return(KirschFilters)

    def CalculateWeightingFunction(HazeImg, Filter, sigma):

        # Computing the weight function... Eq (17) in the paper

        HazeImageDouble = HazeImg.astype(float) / 255.0
        if(len(HazeImg.shape) == 3):
            Red = HazeImageDouble[:, :, 2]
            d_r = circularConvFilt(Red, Filter)

            Green = HazeImageDouble[:, :, 1]
            d_g = circularConvFilt(Green, Filter)

            Blue = HazeImageDouble[:, :, 0]
            d_b = circularConvFilt(Blue, Filter)

            WFun = np.exp(-((d_r**2) + (d_g**2) + (d_b**2)) / (2 * sigma * sigma))
        else:
            d = circularConvFilt(HazeImageDouble, Filter)
            WFun = np.exp(-((d ** 2) + (d ** 2) + (d ** 2)) / (2 * sigma * sigma))
        return(WFun)

    def circularConvFilt(Img, Filter):
        FilterHeight, FilterWidth = Filter.shape
        assert (FilterHeight == FilterWidth), 'Filter must be square in shape --> Height must be same as width'
        assert (FilterHeight % 2 == 1), 'Filter dimension must be a odd number.'

        filterHalsSize = int((FilterHeight - 1)/2)
        rows, cols = Img.shape
        PaddedImg = cv2.copyMakeBorder(Img, filterHalsSize, filterHalsSize, filterHalsSize, filterHalsSize, borderType=cv2.BORDER_WRAP)
        FilteredImg = cv2.filter2D(PaddedImg, -1, Filter)
        Result = FilteredImg[filterHalsSize:rows+filterHalsSize, filterHalsSize:cols+filterHalsSize]

        return(Result)

    ##################
    def psf2otf(psf, shape):
        """
        Convert point-spread function to optical transfer function.
        Compute the Fast Fourier Transform (FFT) of the point-spread
        function (PSF) array and creates the optical transfer function (OTF)
        array that is not influenced by the PSF off-centering.
        By default, the OTF array is the same size as the PSF array.
        To ensure that the OTF is not altered due to PSF off-centering, PSF2OTF
        post-pads the PSF array (down or to the right) with zeros to match
        dimensions specified in OUTSIZE, then circularly shifts the values of
        the PSF array up (or to the left) until the central pixel reaches (1,1)
        position.
        Parameters
        ----------
        psf : `numpy.ndarray`
            PSF array
        shape : int
            Output shape of the OTF array
        Returns
        -------
        otf : `numpy.ndarray`
            OTF array
        Notes
        -----
        Adapted from MATLAB psf2otf function
        """
        if np.all(psf == 0):
            return np.zeros_like(psf)

        inshape = psf.shape
        # Pad the PSF to outsize
        psf = zero_pad(psf, shape, position='corner')

        # Circularly shift OTF so that the 'center' of the PSF is
        # [0,0] element of the array
        for axis, axis_size in enumerate(inshape):
            psf = np.roll(psf, -int(axis_size / 2), axis=axis)

        # Compute the OTF
        otf = np.fft.fft2(psf)

        # Estimate the rough number of operations involved in the FFT
        # and discard the PSF imaginary part if within roundoff error
        # roundoff error  = machine epsilon = sys.float_info.epsilon
        # or np.finfo().eps
        n_ops = np.sum(psf.size * np.log2(psf.shape))
        otf = np.real_if_close(otf, tol=n_ops)

        return otf

    def zero_pad(image, shape, position='corner'):
        """
        Extends image to a certain size with zeros
        Parameters
        ----------
        image: real 2d `numpy.ndarray`
            Input image
        shape: tuple of int
            Desired output shape of the image
        position : str, optional
            The position of the input image in the output one:
                * 'corner'
                    top-left corner (default)
                * 'center'
                    centered
        Returns
        -------
        padded_img: real `numpy.ndarray`
            The zero-padded image
        """
        shape = np.asarray(shape, dtype=int)
        imshape = np.asarray(image.shape, dtype=int)

        if np.alltrue(imshape == shape):
            return image

        if np.any(shape <= 0):
            raise ValueError("ZERO_PAD: null or negative shape given")

        dshape = shape - imshape
        if np.any(dshape < 0):
            raise ValueError("ZERO_PAD: target size smaller than source one")

        pad_img = np.zeros(shape, dtype=image.dtype)

        idx, idy = np.indices(imshape)

        if position == 'center':
            if np.any(dshape % 2 != 0):
                raise ValueError("ZERO_PAD: source and target shapes "
                                "have different parity.")
            offx, offy = dshape // 2
        else:
            offx, offy = (0, 0)

        pad_img[idx + offx, idy + offy] = image

        return pad_img

    #********************************************************************************************8

    def removeHaze(HazeImg, Transmission, A, delta):
        '''
        :param HazeImg: Hazy input image
        :param Transmission: estimated transmission
        :param A: estimated airlight
        :param delta: fineTuning parameter for dehazing --> default = 0.85
        :return: result --> Dehazed image
        '''

        # This function will implement equation(3) in the paper
        # " https://www.cv-foundation.org/openaccess/content_iccv_2013/papers/Meng_Efficient_Image_Dehazing_2013_ICCV_paper.pdf "

        epsilon = 0.0001
        Transmission = pow(np.maximum(abs(Transmission), epsilon), delta)

        HazeCorrectedImage = copy.deepcopy(HazeImg)
        if(len(HazeImg.shape) == 3):
            for ch in range(len(HazeImg.shape)):
                temp = ((HazeImg[:, :, ch].astype(float) - A[ch]) / Transmission) + A[ch]
                temp = np.maximum(np.minimum(temp, 255), 0)
                HazeCorrectedImage[:, :, ch] = temp
        else:
            temp = ((HazeImg.astype(float) - A[0]) / Transmission) + A[0]
            temp = np.maximum(np.minimum(temp, 255), 0)
            HazeCorrectedImage = temp
        return(HazeCorrectedImage)

    #************************************************************************************************

    
    HazeImg = cv2.imread(fileurl)

    # Resize image
    '''
    Channels = cv2.split(HazeImg)
    rows, cols = Channels[0].shape
    HazeImg = cv2.resize(HazeImg, (int(0.4 * cols), int(0.4 * rows)))
    '''

    # Estimate Airlight
    windowSze = 15
    AirlightMethod = 'fast'
    A = Airlight(HazeImg, AirlightMethod, windowSze)

    # Calculate Boundary Constraints
    windowSze = 3
    C0 = 20         # Default value = 20 (as recommended in the paper)
    C1 = 300        # Default value = 300 (as recommended in the paper)
    Transmission = BoundCon(HazeImg, A, C0, C1, windowSze)                  #   Computing the Transmission using equation (7) in the paper

    # Refine estimate of transmission
    regularize_lambda = 1       # Default value = 1 (as recommended in the paper) --> Regularization parameter, the more this  value, the closer to the original patch wise transmission
    sigma = 0.5
    Transmission = CalTransmission(HazeImg, Transmission, regularize_lambda, sigma)     # Using contextual information

    # Perform DeHazing
    HazeCorrectedImg = removeHaze(HazeImg, Transmission, A, 0.85)

    image_save_url = fileurl.replace(filename, "temp_"+filename)

    cv2.imwrite(image_save_url, HazeCorrectedImg)

    # **************************************************************

    image_save_path = "/media/"+"temp_"+filename
    edit_image_url = str(image_save_path)

    return render(request,'image-dehazer.html',
    {'raw_url':templateurl, 'edit_url':edit_image_url})