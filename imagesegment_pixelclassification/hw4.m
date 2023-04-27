% Image Segmentation & Pixel Classification
% Author: David Treder
% I pledge my honor that I have abided by the Stevens Honor System.


function hw4(filename, maskedfilename, unmaskedfilename, testfilename, varargin)
    %part 1 defaults
    defaultTolerance1 = 3;
    defaultNClusters1 = 10;
    
    %part 2 defaults
    defaultScaling2 = 0.8;
    defaultTolerance2 = 3;
    defaultMaxIter2 = 3;
    defaultBlockMode2 = 'crop';
    defaultBlockSize2 = 50;
    defaultLocalShift2 = -1; 
    
    %part 2 defaults
    defaultMaskColor3 = '#fe0000';
    defaultTolerance3 = 1;
    defaultNClusters3 = 10;
    defaultSigma3 = 0;
    
    p = inputParser;
    
    validColor = @(x) ~isnan(matlab.graphics.internal.convertToRGB(x));
    validScalarPosInteger = @(x) isnumeric(x) && isscalar(x) && floor(x) == x && (x>0);
    validScalarNonNegInteger = @(x) isnumeric(x) && isscalar(x) && floor(x) == x && (x>=0);
    validScalarNonNegNum = @(x) isnumeric(x) && isscalar(x) && (x >= 0);
    validScalarPosNum = @(x) isnumeric(x) && isscalar(x) && (x > 0);
    validText = @(x) isstring(x) || ischar(x);
    
    addRequired(p, 'filename1', @isstring);
    addRequired(p, 'filename2', @isstring);
    addRequired(p, 'maskedfilename', @isstring);
    addRequired(p, 'unmaskedfilename', @isstring);
    addRequired(p, 'testfilename', @isstring);
    
    
    addParameter(p, 'tolerance1', defaultTolerance1, validScalarNonNegNum);
    addParameter(p, 'nclusters1', defaultNClusters1, validScalarPosInteger);
    
    addParameter(p, 'scaling2', defaultScaling2, validScalarPosNum);
    addParameter(p, 'tolerance2', defaultTolerance2, validScalarNonNegNum);
    addParameter(p, 'maxiter2', defaultMaxIter2, validScalarPosInteger);
    addParameter(p, 'blockmode2', defaultBlockMode2, validText);
    addParameter(p, 'blocksize2', defaultBlockSize2, validScalarPosInteger);
    addParameter(p, 'localshift2', defaultLocalShift2, validScalarNonNegInteger);
    
    addParameter(p, 'maskcolor3', defaultMaskColor3, validColor);
    addParameter(p, 'tolerance3', defaultTolerance3, validScalarNonNegNum);
    addParameter(p, 'nclusters3', defaultNClusters3, validScalarPosInteger);
    addParameter(p, 'sigma3', defaultSigma3, validScalarNonNegInteger);
    
    parse(p,filename, maskedfilename, unmaskedfilename, testfilename, varargin{:});
      
    S = p.Results;
    sigma3 = S.sigma3;
    testfilename = S.testfilename;
    tolerance2 = S.tolerance2;
    blockmode2 = S.blockmode2;                 
    blocksize2 = S.blocksize2;                 
    filename1 = S.filename1;
    filename2 = S.filename2;
    
    localshift2 = S.localshift2;               
    maskcolor3 = S.maskcolor3;                 
    maskedfilename = S.maskedfilename;         
    maxiter2 = S.maxiter2;                     
    nclusters1 = S.nclusters1;                 
    nclusters3 = S.nclusters3;                 
    scaling2 = S.scaling2;                     
    tolerance1 = S.tolerance1;                 
    tolerance3 = S.tolerance3;                 
    unmaskedfilename = S.unmaskedfilename;
    
    if (localshift2 == -1)
        localshift2 = maxiter2;
    end
    
    image1 = imread(filename1);
    image2 = imread(filename2);
    maskedImage = imread(maskedfilename);
    unmaskedImage = imread(unmaskedfilename);
    testimage = imread(testfilename);
    
    maskcolor3 = uint8(matlab.graphics.internal.convertToRGB(maskcolor3)*255);
    
    %part 1
    [m, n, ~] = size(image1);
    [centroids, clusters] = kmeans(image1, nclusters1, tolerance1);
    newImage = cluster2image(centroids, clusters, m, n);
    figure('name', "K-Means: " + "Clusters: " + nclusters1 + "; Tolerance: " + tolerance1);
    imshow(newImage);
    
    %part 2
    [centroids2, ~, label] = SLIC(image1, blocksize2, scaling2,...
        maxiter2, tolerance2, localshift2, blockmode2);
    newImage = segments2image(centroids2, label);
    figure('name', "SLIC1: " + "Tolerance: " + tolerance2 + ...
        "; Block Size: " + blocksize2 + "; Scaling: " + scaling2 + "; Max Iter: " + maxiter2 + ...
        "; Local Shift: " + localshift2 + "; Block Mode: " + blockmode2);
    imshow(newImage);
    
    [centroids2, ~, label] = SLIC(image2, blocksize2, scaling2,...
        maxiter2, tolerance2, localshift2, blockmode2);
    newImage = segments2image(centroids2, label);
    figure('name', "SLIC2: " + "Tolerance: " + tolerance2 + ...
        "; Block Size: " + blocksize2 + "; Scaling: " + scaling2 + "; Max Iter: " + maxiter2 + ...
        "; Local Shift: " + localshift2 + "; Block Mode: " + blockmode2);
    imshow(newImage);
    
    %part3
    [maskedSet, unmaskedSet] = split_mask(maskedImage, unmaskedImage, maskcolor3);
    maskedSet = reshape(maskedSet, 1, [], 3);
    unmaskedSet = reshape(unmaskedSet, 1, [], 3);

    [colorWords1, ~] = kmeans(maskedSet, nclusters3, tolerance3);
    [colorWords2, ~] = kmeans(unmaskedSet, nclusters3, tolerance3);
    
    if (sigma3 > 0)
        testimageG = imgaussfilt(testimage, sigma3);
    else
        testimageG = testimage;
    end
    classified = classify(testimageG, colorWords1, colorWords2);
    outputImage = paintFromClassified(testimage, classified, maskcolor3, 1);
    figure('name', "Classification: " + "Clusters: " + nclusters3 + "; Tolerance: " + tolerance3 +...
        "; Mask Color: " + mat2str(maskcolor3) + "; Sigma: " + sigma3);
    imshow(outputImage);
end


function image = cluster2image(centroids, clusters, m, n)
    k = size(clusters, 2);
    image = zeros(m, n, 3);
    
    for c = 1:k
        %for each cluster
        mat = cell2mat(clusters(c));
        numPoints = size(mat, 1);
        color = centroids(c, :);
        if (isnan(color(1)))
            %should never really occur hopefully
            disp('nan');
            break;
        end
        for i = 1:numPoints
            row = mat(i, 1);
            col = mat(i, 2);
            image(row, col, :) = color;
        end
    end
    image = cast(image, 'uint8');
    
end

function [centroids, clusters] = kmeans(image, k, tolerance)
    image = cast(image, 'double');
    [m, n, ~] = size(image);
%     disp(k);
%     disp(m);
%     disp(n);
    
    %clusters is a cell array containing k matrices
    %each matrix is to be the list of coordinates
    %the matrix will be a ? by 2 matrix.
    %each line in the matrix will be a point in (r, c) form
    inds = randperm(m*n, k);
    flatimg = reshape(image, m*n, 3);
    centroids = flatimg(inds,:);
    while (true)
        oldCentroids = centroids;        
        clusters =  cell(1, k);
        clusters(1:k) = {zeros(m*n, 2)}; %each cell is a list of points in that group
        clustersC =  cell(1, k);
        clustersC(1:k) = {zeros(m*n, 3)};%each cell is a list of pixel vals for points in that group
        
        
        for i = 1:m
%             disp(i);
            for j = 1:n
                %i, j is a unique pixel in the image
                p1 = reshape(image(i,j,:),1,[]); %unique pixel
                p2 = centroids(1,:); %first val
                minC = 1;
                minDistance = norm(p1 - p2); 
                
                for c = 2:k
                    %c is a unique cluster
                    p2 = centroids(c,:);
                    distance = norm(p1 - p2);
                    if (distance < minDistance)
                        %disp("t");
                        minC = c;
                        minDistance = distance;
                    end
                end
                
                %minC is the cluster with the smallest distance to i, j
                %minDistance is the distance in color space between
                %them(not relevant anymore)
                
                %add i, j to minC
                clusters{minC}(sub2ind([m, n], i, j), :) = [i j];
                clustersC{minC}(sub2ind([m, n], i, j), :) = image(i, j, :);
            end
        end
        
        canFinish=true;
        %recompute vals with current clusters
        for i = 1:k
            a = cell2mat(clusters(i));            
            mat = a(~all(a == 0, 2),:);
            clusters(i) = {mat};
            
            a = cell2mat(clustersC(i));
            mat = a(~all(a == 0, 2),:);
            clustersC(i) = {mat};
            centroids(i,:) = mean(mat, 1);            
        end
        
        for i = 1:k
            
            %If isnan(vals(i,1), that cluster is empty. Split largest
            %cluster using recursive kmeans.
            %If this occurs, force at least one iteration to settle.
            if (isnan(centroids(i,1)) || (size(i, 1) == 0))
%                 disp(clusters);
                
                [~, largestCluster] = max(cellfun(@(x) (size(x, 1)), clusters));
%                 disp(largestCluster);
                largestClusterMatrix = cell2mat(clustersC(largestCluster));
                reshapeLargestCluterMatrix = reshape(largestClusterMatrix, 1, [], 3);
                [splitCentroids, ~] = kmeans(reshapeLargestCluterMatrix, 2, tolerance);
                centroids(i, :) = splitCentroids(2, :);
                centroids(largestCluster, :) = splitCentroids(1, :);
                
                canFinish = false;
                break;%don't want to do it again, wait for next iteration once new seed is filled in
            end
        end
        
        %DEBUG
%         disp(clusters);
%         disp(abs(max(centroids-oldCentroids,[],'all')));
        if (canFinish && abs(max(centroids-oldCentroids,[],'all')) < tolerance)
            break;
        end
    end

end

%part 2d
function image = segments2image(centroids, label)
    [m, n] = size(label);
    image = zeros(m, n, 3);
    label = padImg(label, 1);
    for i = 1:m
        for j = 1:n
            local = label(i:i + 2, j:j + 2);
            if (local == local(1))
                centroid = label(i, j);
                color = centroids(centroid, 3:5);
                image(i, j, :) = reshape(color, [1, 1, 3]);
            else
                image(i, j, :) = zeros(1, 1, 3);
            end
        end
    end
    image = cast(image, 'uint8');
end


function [centroids, clusters, label] = SLIC(image, block_size, scaling, max_iter,...
    tolerance, local_shift, block_mode)

    
    if(~exist('block_mode', 'var'))
        block_mode = 'partial';
    end
    
    if(~exist('local_shift', 'var'))
        local_shift = max_iter;
    end   
    
    [m, n, ~] = size(image);
    switch block_mode
        case 'crop'
            %crop the image such that the dimensions are multiples of
            %block_size
            m = m - (mod(m, block_size));
            n = n - (mod(n, block_size));
            image = image(1:m, 1:n, :);
        case 'warpextend'
            %warps the image such that the dimensions are multiples of
            %block_size, extending dimensions.
            m = m - (mod(m, block_size)) + block_size;
            n = n - (mod(n, block_size)) + block_size;
            image = imresize(image, [m n]);
        case 'warpshrink'
            %warps the image such that the dimensions are multiples of
            %block_size, reducing dimensions.
            m = m - (mod(m, block_size));
            n = n - (mod(n, block_size));
            image = imresize(image, [m n]);
        case 'partial'
            %don't modify image, just have intial blocks be partial. 
    end
    
    image = cast(image, 'double');
    [m, n, ~] = size(image);
    
    block_radius = floor(block_size/2);
    influence = block_size;
    
    label = zeros(m, n);  %cluster the given pixel is assigned to
    distance = Inf(m, n); %distance from assigned cluster
    
    %assume here that image is either a multiple of block_size or blocks
    %can be partial blocks.
    
    rows = (1:block_size:m-block_radius) + block_radius;
    cols = (1:block_size:n-block_radius) + block_radius;
    numCentroids = size(rows, 2)*size(cols, 2);
    centroids = zeros(numCentroids, 5);
    
    c = 1;
    for i = rows
        for j = cols
            centroids(c, :) = [i, j, reshape(image(i, j, :), 1, [])];
            c = c + 1;
        end
    end
    clear c;
    
    
    %compute gradient(pre-part 2)

    gradMagR = myGradMag(image(:, :, 1));
    gradMagG = myGradMag(image(:, :, 2));
    gradMagB = myGradMag(image(:, :, 3));
    
    gradMag = sqrt(gradMagR.^2 + gradMagG.^2 +gradMagB.^2);
    
%     gradMag = imgGradMag(sobel(image, 'h'), sobel(image, 'v'));
    
    for iter = 1:max_iter %part 5
        oldCentroids = centroids;
        
        %local shift (part 2)
        
        if (iter <= local_shift)
            for c = 1:numCentroids
                i = floor(centroids(c, 1));
                j = floor(centroids(c, 2));

                local = gradMag(i - 1:i + 1, j - 1:j + 1);
                [~, linearOffset] = min(local, [], 'all', 'linear');
                [iOffset, jOffset] = ind2sub(size(local), linearOffset);
                iOffset = iOffset - floor(size(local, 1)/2);
                jOffset = jOffset - floor(size(local, 1)/2);
                newI = iOffset + i;
                newJ = jOffset + j;
                newColor = reshape(image(newI, newJ, :), 1, []);

                centroids(c, :) = [newI, newJ, newColor];        
            end
        end
        
        %3/4 -  Localized Centroid Update
        %For every centroid, loop through the region in a block_size*2 by
        %block_size*2 region
        
        for c = 1:numCentroids
            centroid = floor(centroids(c, :));
            row = centroid(1);
            col = centroid(2);
            centroid(1:2) = centroid(1:2)/scaling;
            
            ranges = [row-influence row+influence col-influence col+influence];
            ranges = arrayfun(@(x) (x<1) + (x>=1)*x, ranges); %if <1 set to 1, otherwise keep
            if (ranges(1) > m)
                ranges(1) = m;
            end
            if (ranges(2) > m)
                ranges(2) = m;
            end
            if (ranges(3) > n)
                ranges(3) = n;
            end
            if (ranges(4) > n)
                ranges(4) = n;
            end
            

            for i = ranges(1):ranges(2)
                for j = ranges(3):ranges(4)
                    %i, j is unique pixel in infuence of c
                    pixel = [i, j, reshape(image(i, j, :), 1, [])];
                    pixel = cast(pixel, 'double');
                    pixel(1:2) = pixel(1:2)/scaling;
                    D = norm(pixel-centroid);
                    if (D < distance(i, j))
                        distance(i, j) = D;
                        label(i, j) = c;
                    end
                end
            end

        end
        
        %finished an iteration, now recompute and decide to loop again

        
        clusters =  cell(1, numCentroids);
        clusters(1:numCentroids) = {zeros(block_size*block_size, 5)}; %each cell is a list of points in that group
        nextIndex = ones(1, numCentroids);
        for i = 1:m
            for j = 1:n
                newCluster = [i, j, reshape(image(i, j, :), 1, [])];
                clusters{label(i, j)}(nextIndex(label(i, j)), :) = newCluster;
                nextIndex(label(i, j)) = nextIndex(label(i, j)) + 1;
            end
%             disp(i);
        end
        
        for i = 1:numCentroids
            a = cell2mat(clusters(i));
            mat = a(~all(a == 0, 2),:);
            clusters(i) = {mat};
            
            centroids(i,:) = mean(mat, 1);
        end
        
        if (abs(max(centroids-oldCentroids,[],'all')) < tolerance)
            break;
        end
% 
%         disp('end loop');
%         disp(iter);
    end
end


%part 3

function [maskedSet, unmaskedSet] = split_mask(maskedImage, unmaskedImage, maskColor)
    %given two images, one with a mask applied, returns 2 dimensional
    %matrices of the set of pixels that are masked and the set of pixels that
    %were not masked. matrices are in the form 3 by ?
    
    [m, n, ~] = size(maskedImage);
    maskedSet = zeros(m*n, 3)-1;
    unmaskedSet = zeros(m*n, 3)-1;
    maskColor = reshape(maskColor, 1, 1, 3);
    for i = 1:m
        for j = 1:n
            maskPixel = maskedImage(i, j, :);
            unmaskPixel = unmaskedImage(i, j, :);
            if (isequal(maskPixel, maskColor))
                %pixel is part of maskedset
                maskedSet(sub2ind([m, n], i, j), :) = unmaskPixel;
            else
                %pixel is not part of maskedset
                unmaskedSet(sub2ind([m, n], i, j), :) = unmaskPixel;
            end
        end
    end
    maskedSet = maskedSet(~all(maskedSet == -1, 2),:);
    unmaskedSet = unmaskedSet(~all(unmaskedSet == -1, 2),:);
end

function classified = classify(image, set1, set2)
    %given an image and 2 sets of color visual words, iterates through the
    %image and classifies each pixel as part of set1 or set2. returns
    %classified, a matrix with the same dimensions as image, containing a 
    %1 or 2 based on the set the pixel is a part of.
    
    [m, n, ~] = size(image);
    [~, k1] = size(set1);
    [~, k2] = size(set2);
    classified = zeros(m, n);
    
    image = cast(image, 'double');
    for i = 1:m
        for j = 1:n
            p1 = reshape(image(i,j,:),1,[]); %unique pixel
            p2 = set1(1,:); %first val
            minDistance1 = norm(p1-p2);
            for c = 2:k1
                p2 = set1(c,:);
                distance = norm(p1 - p2);
                if (distance < minDistance1)
                    minDistance1 = distance;
                end
            end
            
            p2 = set2(1,:); %first val
            minDistance2 = norm(p1-p2);
            for c = 2:k2
                p2 = set2(c,:);
                distance = norm(p1 - p2);
                if (distance < minDistance2)
                    minDistance2 = distance;
                end
            end
            
            if (minDistance1 < minDistance2)
                classified(i, j) = 1;
            else
                classified(i, j) = 2;
            end
                
            
        end
    end
    
end

function paintedImage = paintFromClassified(image, classified, color, set)
    %given an image, paints over the image with color where the
    %corresponding pixel in classified == set
   
    [m, n, ~] = size(image);
    paintedImage = image;
    for i = 1:m
        for j = 1:n
            if (classified(i, j) == set)
                paintedImage(i, j,:) = color;
            end
        end
    end
end


%FROM PREVIOUS ASSIGNMENTS

function padded = padImg(image, p)
    %given an image and an integer, returns the image with the appropriate
    %padding
    
    %step 0: assign variables to useful values
    [m,n] = size(image); %dimensions of original image
    %step 1: create new matrix with correct size
    padded = zeros(m+p+p, n+p+p, 'double');

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

%imshow(image);
function pixelVal = step(window, filter)
    temp = cast(window, 'double').*filter;
    pixelVal = sum(temp, 'all');
end

%modified from hw function
function filteredImg = fImg(image, filter)
    %given an image and a filter, returns the filtered image.
    %uses replicate padding
    
    %filter size
    [s,~] = size(filter);
    %original image size (unpadded)
    [m,n] = size(image);
    %padding
    p = floor(s/2);
    
    filteredImg = zeros(m, n, 'double');
    paddedImage = padImg(image, p);
    
    for i = 1:m
        for j = 1:n
            i_p = i + p;
            j_p = j + p;
            window = paddedImage(i_p - p:i_p + p, j_p - p:j_p + p);
            filteredImg(i,j) = step(window, filter);
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


function filteredImg = gauss(image, sigma)   
    f = generateGauss(sigma, 1+6*round(sigma));
    f = f/(sum(f, 'all')); %normalizing constant
    filteredImg = fImg(image, f);
end

function filteredImg = sobel(image, dir)
    vSobel = [1 0 -1; 2 0 -2; 1 0 -1];
    hSobel = [1 2 1; 0 0 0; -1 -2 -1];
    switch dir
        case 'v'
            filteredImg = fImg(image, vSobel);
        case 'h'
            filteredImg = fImg(image, hSobel);
    end
end

function [h, v] = gaussSobel(image, sigma)
    g = gauss(image, sigma);
    h = sobel(g, 'h');
    v = sobel(g, 'v');
end

function gradmag = myGradMag(image)
    hfiltered = sobel(image, 'h');
    vfiltered = sobel(image, 'v');
    
    gradmag = (imgGradMag(vfiltered, hfiltered));
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


















