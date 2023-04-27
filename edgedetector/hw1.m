% Edge detection
% Author: David Treder
% I pledge my honor that I have abided by the Stevens Honor System.



function hw1(filename, sigma, cutoff) %main function 
    
    image = imread(filename);
    
    [vfiltered, hfiltered] = gaussSobel(image, sigma);
    
    gradmag = (imgGradMag(vfiltered, hfiltered));
    grad = imgGrad(vfiltered, hfiltered);
    
    culled = cull(gradmag, cutoff);
    
    final = nonMaxSupp(culled, grad);
    
    %montage([vfiltered image gradmag culled final]);
    
    imshow(cast(final, 'uint8'));
end

function nms = nonMaxSupp(gradMagMatrix, gradMatrix)
    
    gradMagMatrix = padImg(gradMagMatrix, 1, 'double');
    gradMatrix = padImg(gradMatrix, 1, 'double');
    
    [m,n] = size(gradMagMatrix);
    disp(m);
    disp(n);
    
    nms = zeros(m,n, 'double');
    
    for i = 2:m-1
        for j = 2:n-1
            curr = gradMatrix(i, j);
            curr = round(4*(curr/pi));
            pCurr = gradMagMatrix(i, j);
            p1 = 0;
            p2 = 0;
            switch curr
                case 0
                    %vertical
                    p1 = gradMagMatrix(i, j+1);
                    p2 = gradMagMatrix(i, j-1);
                case 1
                    %+diagonal
                    p1 = gradMagMatrix(i+1, j+1);
                    p2 = gradMagMatrix(i-1, j-1);
                    
                case {-2, 2}
                    %horizontal
                    p1 = gradMagMatrix(i+1, j);
                    p2 = gradMagMatrix(i-1, j);
                case -1
                    %-diagonal
                    p1 = gradMagMatrix(i-1, j+1);
                    p2 = gradMagMatrix(i+1, j-1);
                    
                    
            end
            
            if (pCurr > p1 && pCurr > p2)
                %larger than neighbor pixels
                %should be kept.
                nms(i, j) = 255;
                %default is black.
                
                
                %Testing code that color codes the output image based on
                %what the detected edge direction is.
%                 if (curr == -2)
%                     curr = 2;
%                 end
%                 nms(i,j) = (2+curr)*70;
            end
            
        end
    end
    nms = nms(2:m-1, 2:n-1);
    
end

function gMagMatrix = imgGradMag(vImg, hImg)
    [m,n] = size(vImg);
    gMagMatrix = zeros(m, n);
    for i = 1:m
        for j = 1:n
            vCurr = vImg(i, j);
            hCurr = hImg(i, j);
            gMagMatrix(i, j) = sqrt(vCurr*vCurr + hCurr*hCurr);
        end
    end
end

function gMatrix = imgGrad(vImg, hImg)
    [m,n] = size(vImg);
    gMatrix = zeros(m, n);
    for i = 1:m
        for j = 1:n
            num = double(vImg(i, j));
            den = double(hImg(i, j));
            temp  = num/den;
            temp2 = atan(temp);
            if (isnan(temp2))
                temp2 = 0;
            end
            gMatrix(i, j) = temp2;
        end
    end
    
end

function image = cull(image, min)
    [m,n] = size(image);
    for i = 1:m
        for j = 1:n
            pixel = image(i, j);
            if (pixel < min)
                image(i, j) = 0;
            else
                %image(i, j) = 255;
            end
            
        end
    end
    
end

function gfilter = generateGauss(sigma, s)
    gfilter = zeros(s, s);
    
    for i = 1:s
        for j = 1:s
            x = i - floor(s/2)-1;
            y = j - floor(s/2)-1;
            denom = 2*pi*sigma*sigma;
            numer = exp(-1* ((x*x+y*y)/(2*sigma*sigma)));
            
            gfilter(i, j) = numer/denom;
        end
    end
    
    
    
end

function padded = padImg(image, p, type)
    %given an image and an integer, returns the image with the appropriate
    %padding
    
    %step 0: assign variables to useful values
    [m,n] = size(image); %dimensions of original image
    %step 1: create new matrix with correct size
    padded = zeros(m+p+p, n+p+p, type);

    %step 2: copy old matrix to new matrix (edges still 0s)
    padded(p+1:p+m, p+1:p+n) = image; 
    
    %step 3: extend edges    
    for i = 1:p
        padded(i:i, 1:n+p+p) = padded(p+1:p+1, 1:n+p+p);
    end
    
    for i = p+m:m+p+p
        padded(i:i, 1:n+p+p) = padded(p+m:p+m, 1:n+p+p);
    end
    
    for i = 1:p
        padded(1:m+p+p, i:i) = padded(1:m+p+p, p+1:p+1);
    end
    
    for i = p+n:n+p+p
        padded(1:m+p+p, i:i) = padded(1:m+p+p, p+n:p+n);
    end
    
end

function pixelVal = step(window, filter)
    temp = cast(window, 'double').*filter;
    pixelVal = sum(temp, 'all');
end

function filteredImg = fImg(image, filter, type)
    %given an image and a filter, returns the filtered image.
    %uses replicate padding
    
    %filter size
    [s,~] = size(filter);
    %original image size (unpadded)
    [m,n] = size(image);
    %padding
    p = floor(s/2);
    
    filteredImg = zeros(m, n, type);
    paddedImage = padImg(image, p, 'uint8');
    
    for i = 1:m
        for j = 1:n
            i_p = i + p;
            j_p = j + p;
            window = paddedImage(i_p - p:i_p + p, j_p - p:j_p + p);
            filteredImg(i,j) = step(window, filter);
        end
    end
end

function filteredImg = gauss(image, sigma)   
    f = generateGauss(sigma, 1+10*round(sigma)); %10 ensures filter sums to 1
    filteredImg = fImg(image, f, 'double');
end

function filteredImg = sobel(image, dir)
    vSobel = [1 0 -1; 2 0 -2; 1 0 -1];

    hSobel = [1 2 1; 0 0 0; -1 -2 -1];
    switch dir
        case 'v'
            filteredImg = fImg(image, vSobel, 'double');
        case 'h'
            filteredImg = fImg(image, hSobel, 'double');
    end
end

function [h, v] = gaussSobel(image, sigma)
    g = gauss(image, sigma);
    h = sobel(g, 'h');
    v = sobel(g, 'v');
end


