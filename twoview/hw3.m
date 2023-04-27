% Image Alignment
% Author: David Treder
% I pledge my honor that I have abided by the Stevens Honor System.

function hw3(filename1, filename2, numFeatures, alpha, sigma, topPoints, method,...
    windowRadius, randPoints, leftBright, rightBright,...
    inliers_a1, distance_a1, maxIterations_a1, ...
    inliers_a2, distance_a2, maxIterations_a2)
    
    %main function
    %outputs everything for the assignment specified in the pdf
    %figures are titled based on the problem part

    if(~exist('numFeatures', 'var'))
        numFeatures = 20;
    end
    
    if(~exist('alpha', 'var'))
        alpha = 0.05;
    end
    
    if(~exist('sigma', 'var'))
        sigma = 3;
    end
    
    if(~exist('topPoints', 'var'))
        topPoints = 1000;
    end
    
    if(~exist('method', 'var'))
        method = 'ncc';
    end
    
    if(~exist('windowRadius', 'var'))
        windowRadius = 40;
    end
    
    if(~exist('randPoints', 'var'))
        randPoints = 30;
    end
    
    if(~exist('leftBright', 'var'))
        leftBright = .5;
    end
    
    if(~exist('rightBright', 'var'))
        rightBright = .84;
    end
    
    if(~exist('inliers_a1', 'var'))
        inliers_a1 = 17;
    end
    
    if(~exist('distance_a1', 'var'))
        distance_a1 = 6;
    end
    
    if(~exist('maxIterations_a1', 'var'))
        maxIterations_a1 = 100000;
    end
    
     if(~exist('inliers_a2', 'var'))
        inliers_a2 = inliers_a1;
    end
    
    if(~exist('distance_a2', 'var'))
        distance_a2 = distance_a1;
    end
    
    if(~exist('maxIterations_a2', 'var'))
        maxIterations_a2 = maxIterations_a1;
    end
    

    close all;
    if (strcmp(method, 'ncc'))
        order = 'descend';
    else
        order = 'ascend';
    end
    
    
    image1c = imread(filename1);
    image2c = imread(filename2);
    image1 = rgb2gray(image1c);
    image2 = rgb2gray(image2c);
    
    [y,x] = size(image1);
    
    
    %1a start
    %image1
    [detMatrix, traceMatrix] = SMM(image1, sigma);
    R1 = cornerResponse(detMatrix, traceMatrix, alpha);
    R_t1 = cullTop(R1, topPoints);
    
    %image2
    [detMatrix, traceMatrix] = SMM(image2, sigma);
    R2 = cornerResponse(detMatrix, traceMatrix, alpha);

    R_t2 = cullTop(R2, topPoints);

    %1a output
    figure('name', '1a: Top 1000 Points');
    montage([R_t1, R_t2]);
    
    %1b start
    [nms1, points1] = nonMaxSupp(R_t1);
    [nms2, points2] = nonMaxSupp(R_t2);
    
    %1b output
    figure('name', '1b: Top 1000 Points with Non-Maximum Suppression');
    montage([nms1, nms2]);
    
    %1c start
    simVector = computeSimilarities(points1, image1, points2, image2, windowRadius, method, order);

    [topPoints1, topPoints2] = getTopPoints(simVector, numFeatures);
    
    figure('name', "1c: Top " + numFeatures + " features matched");
    displayMatchedFeatures(image1c, image2c, topPoints1, topPoints2);
    
    %1d start
    rotated_image1c = imrotate(image1c, 45);
    rotated_image1 = rgb2gray(rotated_image1c);
    
    [detMatrix, traceMatrix] = SMM(rotated_image1, sigma);
    R1 = cornerResponse(detMatrix, traceMatrix, alpha);
    R_t1 = cullTop(R1, topPoints);
    
    [nms1_45, points1_45] = nonMaxSupp(R_t1);
    
    simVector_45 = computeSimilarities(points1_45, rotated_image1, points2, image2, windowRadius, method, order);
    [topPoints1_45, topPoints2_45] = getTopPoints(simVector_45, numFeatures);
    
    figure('name', "1d 45deg: Top " + numFeatures + " features matched");
    displayMatchedFeatures(rotated_image1c, image2c, topPoints1_45, topPoints2_45);
    
    %1d with 20
    
    rotated_image1c = imrotate(image1c, 10);
    rotated_image1 = rgb2gray(rotated_image1c);
    
    [detMatrix, traceMatrix] = SMM(rotated_image1, sigma);
    R1 = cornerResponse(detMatrix, traceMatrix, alpha);
    R_t1 = cullTop(R1, topPoints);
    
    [nms1_10, points1_10] = nonMaxSupp(R_t1);
    
    simVector_10 = computeSimilarities(points1_10, rotated_image1, points2, image2, windowRadius, method, order);
    [topPoints1_10, topPoints2_10] = getTopPoints(simVector_10, numFeatures);
    
    figure('name', "1d 10 deg: Top " + numFeatures + " features matched");
    displayMatchedFeatures(rotated_image1c, image2c, topPoints1_10, topPoints2_10);
    
    
    %END OF PROBLEM 1
    %START OF PROBLEM 2
    
    %2a start
    a1 = horzcat(topPoints1, topPoints2);
    
    c = randperm(length(simVector), randPoints);
    a2 = simVector(c,1:4);
    
    %2b start
    [transform_a1, inliers_a1, i_a1] = ransac(a1, inliers_a1, distance_a1, maxIterations_a1);
    [transform_a2, inliers_a2, i_a2] = ransac([a2;a1], inliers_a2, distance_a2, maxIterations_a2);
    
    if (i_a1 == maxIterations_a1 || i_a2 == maxIterations_a2)
        disp("exiting main");
        return;
    end
    

    left_a1 = inliers_a1(:, 1:2);
    right_a1 = inliers_a1(:, 3:4);
    left_a2 = inliers_a2(:, 1:2);
    right_a2 = inliers_a2(:, 3:4);
    
%     [left_a1, right_a1] = getTopPoints(inliers_a1, size(inliers_a1, 1));
%     [left_a2, right_a2] = getTopPoints(inliers_a2, size(inliers_a2, 1));
    
    figure('name', "2b: Inliers with RANSAC: a1");
    displayMatchedFeatures(image1c, image2c, left_a1, right_a1);
    figure('name', "2b: Inliers with RANSAC: a2");
    displayMatchedFeatures(image1c, image2c, left_a2, right_a2);
    
    distances_a1 = inliers_a1(:, 5);
    distances_a2 = inliers_a2(:, 5);
    avg_a1 = mean(distances_a1);
    avg_a2 = mean(distances_a2);
    
    disp("Number of iterations a1: " + i_a1);
    disp("Number of iterations a2: " + i_a2);
    
    
    disp("Average reprojection error a1: "+avg_a1);
    disp("Average reprojection error a1 and a2: "+avg_a2);
    
    %2c start
    [image1cT_a1, xdata_a1, ydata_a1] = applyTransform(image1c, transform_a1);
    
    [image1cT_a2, xdata_a2, ydata_a2] = applyTransform(image1c, transform_a2);
    
    pan_a1 = combineImages(image1cT_a1, image2c, xdata_a1, ydata_a1);
    pan_a2 = combineImages(image1cT_a2, image2c, xdata_a2, ydata_a2);
    
    figure('name', '2c: Combined panorama: a1');
    imshow(pan_a1);
    figure('name', '2c: Combined panorama: a2');
    imshow(pan_a2);
    
    %2d

    
    dim_left = shadeImage(image1cT_a1, leftBright, rightBright);
    pan_a1_dim = combineImages(dim_left, image2c, xdata_a1, ydata_a1);
    figure('name', '2d: Combined panorama with color correction: a1');
    imshow(pan_a1_dim);
end

function dim_image = shadeImage(image, min, max)
    dim = linspace(min,max,size(image, 2));
    dim = repmat(dim,[size(image, 1) 1 3]);
    dim_image = uint8(dim.*single(image)); 

end

function image2pad = combineImages(image1, image2, xdata, ydata)
    [ty, ~, ~] = size(image2);

    image2pad = horzcat(zeros(ty, floor(xdata(1) * -1), 3), image2);
    [~, tx, ~] = size(image2pad);
    image2pad = vertcat(zeros( floor(ydata(1) * -1), tx, 3), image2pad);

    [ty, tx, ~] = size(image1);

    for i = 1:ty
        for j = 1:tx
            pix1 = reshape(image2pad(i, j, :), 1, []);
            pix2 = reshape(image1(i, j, :), 1, []);
            zro = [0, 0, 0];
            if (isequal(pix1, zro) || isequal(pix2, zro))
                image2pad(i, j, :) = image2pad(i, j, :) + image1(i, j, :);
            else
                image2pad(i, j, :) = (image2pad(i, j, :)/2 + image1(i, j, :)/2);
            end
        end
    end
end

function [im, xdata, ydata] = applyTransform(image, transform)
    t = (reshape(transform, [3, 2]));
    t = horzcat(t, [0;0;1]);
    tform = maketform('affine', t);

    %using nearest neighbor interpolation for better combining
    [im, xdata, ydata] = imtransform(image, tform, 'nearest');
end

function R = cornerResponse(detMatrix, traceMatrix, a)
    R = detMatrix - a * (traceMatrix.^2);
end

function [detMatrix, traceMatrix] = SMM(image, sigmaM)
    % return:
    % matrix of determinants of M [detMatrix]
    % matrix of trace of M [traceMatrix]
    
    [m, n] = size(image);
    detMatrix = zeros(m, n);
    traceMatrix = zeros(m, n);
    
    
    
    [hSobel, vSobel] = gaussSobel(image, sigmaM);
    
    Ixx = sobel(vSobel, 'v');
    Iyy = sobel(hSobel, 'h');
    Ixy = sobel(hSobel, 'v');
    
    for i = 1:m
        for j = 1:n
            M = [Ixx(i,j) Ixy(i,j) ; Ixy(i,j) Iyy(i,j)];
            detMatrix(i, j) = det(M);
            traceMatrix(i, j) = trace(M);
        end
    end
end

function ncc = nccSim(p1, p2)
    p1 = p1(:);
    p2 = p2(:);
    
    p1Mean = mean(p1, 'all');
    p2Mean = mean(p2, 'all');
    
    
    
    num = sum((p1 - p1Mean).*(p2 - p2Mean));
    den = sqrt(sum((p1 - p1Mean).^2)*sum((p2 - p2Mean).^2));
    
    ncc = num/den;
    
    
end

function ssd = ssdSim(p1, p2)
    %given two square matrixes of the same size
    %compute ssd similarity measure
    
    X=p1-p2;
    ssd = sum(X(:).^2); 
end

function m = computeSimilarities(points1, im1, points2, im2, r, mode, order)
    %given two point matrixes and two images
    %computes the similarities of the patches around each point
    %returns an ordered matrix of the similarities
    %matrix is in form im1y im1x im2y im2x sim
    
    %r parameter determines radius of patch
    %r of 2 will use 5x5 patch
    
    %mode parameter decides what similarity measure should be used.
    %default is SSD
    
    if(~exist('mode', 'var'))
        mode = 'ssd';
    end
    
    switch mode
        case 'ssd'
            sf = @ssdSim;
        case 'ncc'
            sf = @nccSim;
    end
    
    %
    
    im1 = padImg(im1, r);
    im2 = padImg(im2, r);
    
    
    [p1size, ~] = size(points1);
    [p2size, ~] = size(points2);
    
    m = zeros(p1size*p2size, 5);
    
    for i = 1:p1size
        point1 = points1(i,:);
        patch1 = im1( point1(1)-r+r:point1(1)+r+r , point1(2)-r+r:point1(2)+r+r );
        for j = 1:p2size
            
            point2 = points2(j,:);
            patch2 = im2(point2(1)-r+r:point2(1)+r+r, point2(2)-r+r:point2(2)+r+r);
            
            sim = sf(patch1, patch2);
            
            m(sub2ind([p1size, p2size],i,j),:) = [point1 point2 sim];
            
        end
    end
    m = sortrows(m, 5, order);
end

function [points1, points2] = getTopPoints(simVector, num)
    %given a simVector, which is a n by 5 vector containing two points and
    %a similarity score, returns the first [num] pairs of points, stored in
    %separate num by 2 vectors
    
    points1 = simVector(1:num, 1:2);
    points2 = simVector(1:num, 3:4);
    
end

function displayMatchedFeatures(image1, image2, points1, points2)
    %displays matched features in the provided images
    %functions very similarly to showMatchedFeatures from computer vision
    %toolbox
    
    showMatchedFeatures(image1, image2, fliplr(points1), fliplr(points2), 'montage');
    
end

function final = cullTop(image, num)
    %returns a filtered image with only the top num values
    
    [m, n] = size(image);
    final = zeros(m, n);
    
    for q = 1:num
    
        maxi = 1;
        maxj = 1;

        for i = 1:m
            for j = 1:n
                if (image(i, j) > image(maxi, maxj))
                    maxi = i;
                    maxj = j;
                end
            end
        end
        final(maxi, maxj) = image(maxi, maxj);
        image(maxi, maxj) = -Inf;
    end
end

%%%%
%functions for part 2
%%%%

function [transform, inliers, i] = ransac(featurePairs, c,d,n)
    %applies ransac on the dataset given by featurePairs
    %returns the affine transformation parameters of a transformation with
    %strong support
    %runs at most n times
    %"strong support" defined as having >= c inliers
    %inliers being points <= d distance from transformed point
    
    
    for i = 1:n
        [s, ~] = size(featurePairs);
        r = randperm(s, 3);
        fp1 = featurePairs(r(1), :);
        fp2 = featurePairs(r(2), :);
        fp3 = featurePairs(r(3), :);
        
        transform = affine([fp1;fp2;fp3]);
        [inliers, ~] = fitToTransform(featurePairs,transform,d);
        [nIn, ~] = size(inliers);
%         if (nIn >0)
%             disp(nIn);
%         end
        if (nIn >= c)
            %found good transform
            %return to calling function
            return;
        end
        
    end
    transform = 0;
    inliers = [];
    disp("ERROR: Unable to find an acceptable fit in " + n + " iterations.");
    disp("Consider allowing more iterations, or relax requirements for a transform.");
end

function [inliers, outliers] = fitToTransform(featurePairs, transform, d)
    %given a matrix of featurePairs
    %   a transform
    %   and a distance d
    %returns the inlier and outlier feature pairs
    %where an inlier is a feature pair where when the first point is
    %transformed, the distance from the second point is <= d
    
    [n, ~] = size(featurePairs);
    inliers = horzcat(featurePairs, zeros(n, 1));
    outliers = horzcat(featurePairs, zeros(n, 1));
    blk = zeros(1,5); %row to blank out a pair that is not in the correct matrix
    
    [n, ~] = size(featurePairs);
    
    for i = 1:n
        p = featurePairs(i,:);
        c = num2cell(p);
        [y1, x1, y2, x2] = c{:};
        
        [y, x] = applyTransformation(transform, [y1, x1]);
%         disp([y, x]);
%         disp([y1, x1]);
%         disp([y2, x2]);
        
        if (norm([y, x] - [y2, x2]) <= d) %distance between calculated and given point
            %inlier!
            outliers(i, :) = blk;
            inliers(i, 5) = norm([y, x] - [y2, x2]);
        else
            inliers(i, :) = blk;
            outliers(i, 5) = norm([y, x] - [y2, x2]);
        end
    end
    
    %now all points that are incorrect in each matrix is 0,0,0,0
    %remove all 0,0,0,0 rows from matrix
    inliers = inliers(~all(inliers == 0, 2),:);
    outliers = outliers(~all(outliers == 0, 2),:);
    
end

function an = affine(points)
    %given a 3 by 4 matrix of point pairings
    %returns the affine transformation matrix
    %point pairings are given in the form
    %[y1, x1, y1_, x1_]
    
    c = num2cell(points);
    [y1, y2, y3, x1, x2, x3, y1_, y2_, y3_, x1_, x2_, x3_] = c{:};
    
    %area of triangle made of the points.
    a = (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2));
    a_ = (x1_ * (y2_ - y3_) + x2_ * (y3_ - y1_) + x3_ * (y1_ - y2_));
    if (a == 0 || a_ == 0)
        an = [0 0 0 0 0 0];
        return;
    end
    
    A = [x1 y1 1 0 0 0;
         0 0 0 x1 y1 0;
         x2 y2 1 0 0 0;
         0 0 0 x2 y2 1;
         x3 y3 1 0 0 0;
         0 0 0 x3 y3 1];
    
    b = [x1_; y1_; x2_; y2_; x3_; y3_];
    
    an = linsolve(A, b);
    
end

function [y, x] = applyTransformation(t, point)
    %given a matrix with 6 values t that defines a transformation
    %and a point in the form [y, x]
    %returns the new y and x positions
    
    t = transpose(reshape(t, [3, 2]));
    point = [point(2) point(1) 1];
    point = reshape(point, [3,1]);
%     disp(point);
%     disp(t);
    a = t*point;
    x = a(1);
    y = a(2);
end
%%%%
%function from prev assignments
%%%%

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

function [nms, points] = nonMaxSupp(image)
    %applies 3x3 nms to the image
    %returns nms image and matrix of points
    
    [m,n] = size(image);
    
%     disp(m);
%     disp(n);

    image = padImg(image, 1);
    
    nms = zeros(m,n, 'double');
    points = zeros(0, 2);
    
    for i = 2:m-1
        for j = 2:n-1
            local = image(i - 1:i + 1, j - 1:j + 1);
            %disp(local);
            
            %if the curr pixel is a nonzero maximum
            if (image(i, j) ~= 0 && max(local, [], 'all') == image(i,j))
               nms(i, j) = image(i, j);
               nms(i,j) = 255;
               points = [points;[i j]];
            end            
        end
    end
    
    %nms = nms(2:m-1, 2:n-1);
    
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


