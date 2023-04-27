% Line detection (2 ways)
% Author: David Treder
% I pledge my honor that I have abided by the Stevens Honor System.


function hw2(filename, distance, count, iterLimit, binThetaSize, binRhoSize,...
    sigmaR, thresholdR, noOfLinesR, sigmaH, thresholdH, noOfLinesH)
    %main function of the program
    %user can specify the parameters of the program
    %separate parameters for hough and ransac, unless the hough parameters
    %are omitted.
    
    %defaut values are used if the argument is omitted .
    
    close all;
    
    if(~exist('distance', 'var'))
        distance = 3;
    end
    
    if(~exist('count', 'var'))
        count = 55;
    end
    
    if(~exist('iterLimit', 'var'))
        iterLimit = 45000;
    end
    
    if(~exist('binThetaSize', 'var'))
        binThetaSize = 2;
    end
    
    if(~exist('binRhoSize', 'var'))
        binRhoSize = 3;
    end
    
    if(~exist('sigmaR', 'var'))
        sigmaR = 2;
    end
    
    if (~exist('nOfLinesR', 'var'))
        noOfLinesR = 4;
    end
    
    if (~exist('thresholdR', 'var'))
        thresholdR = 8000;
    end
    
    if (~exist('sigmaH', 'var'))
        sigmaH = sigmaR;
    end
    
    if (~exist('nOfLinesH', 'var'))
        noOfLinesH = noOfLinesR;
    end
    
    if (~exist('thresholdH', 'var'))
        thresholdH = thresholdR;
    end
    
    function s = varSummaryR
        s = "Sigma: "+sigmaR+"; Threshold: "+thresholdR+...
            "; Lines: "+noOfLinesR+"; Distance: "+distance+...
            "; Count: "+count+"; Iterations: "+iterLimit;
    end

    function s = varSummaryH
        s = "Sigma: "+sigmaH+"; Threshold: "+thresholdH+...
            "; Lines: "+noOfLinesH+"; Theta Bin Dim: "+binThetaSize+...
            "; Rho Bin Dim: "+binRhoSize;
    end
    
    image = imread(filename);
    [y,x] = size(image);
    
    %RANSAC
    [hS, vS] = gaussSobel(image, sigmaR);
    hessianMatrix = applyHessian(vS, hS);
    
    absH = abs(hessianMatrix);
    culled = cull(absH, thresholdR);
    [nms, points] = nonMaxSupp(culled);
    

    [lines, inliers] = multipleRansac(points, iterLimit, distance, count, noOfLinesR);
    figure('name', 'RANSAC Image: '+varSummaryR());
    imshow(image);
    hold on;
    drawLines(lines);
    plotPoints(inliers);
    hold off;
    
    figure('name', 'RANSAC Points: '+varSummaryR());
    imshow(nms);
    hold on;
    drawLines(lines);
    plotPoints(inliers);
    hold off;
    
    %Hough
    rhoMax = max(x, y);
    
    [hS, vS] = gaussSobel(image, sigmaH);
    hessianMatrix = applyHessian(vS, hS);
    
    absH = abs(hessianMatrix);
    culled = cull(absH, thresholdH);
    [nms, points] = nonMaxSupp(culled);
    
    houghMatrix = hough(points, binRhoSize, binThetaSize, rhoMax);
    houghMatrix = nonMaxSupp(houghMatrix);
    
    [lines, inliers] = multipleFindMaxSupportLine(houghMatrix, binRhoSize, binThetaSize, rhoMax, points, noOfLinesH);

    figure('name', 'Hough Image: '+varSummaryH());
    imshow(image);
    hold on;
    drawLines(lines);
    plotPoints(inliers);
    hold off;
    
    figure('name', 'Hough Points: '+varSummaryH());
    imshow(nms);
    hold on;
    drawLines(lines);
    plotPoints(inliers);
    hold off;
end

function result = pointVotedInBin(px, py, binRhoSize, binThetaSize, binTheta, binRho, maxRho)
    min = (binTheta-1)*binThetaSize;
    max = binTheta*binThetaSize;
    
    if (ceil(min) == min)
        min = min+1;
    end
    
    min = ceil(min);
    max = floor(max);
    thetaV = min:max;
    [~,n] = size(thetaV);
    %disp(thetaV);
    for i = 1:n
        t = thetaV(i);
        r = (px * cosd(t) + py * sind(t));
        r = r + maxRho + 1;
        r = ceil(r/binRhoSize);
        
        if (r == binRho)
            result = true;
            return;
        end
        
    end
    result = false;
    
end

function [mx1, mx2, my1, my2] = extremes(points)
    %returns the coordinates of the extreme points of the inliers
    [n,~] = size(points);
    %disp(n);
    maxd = 0;
    for i = 1:n
        for j = 1:n
            p1 = points(i,:);
            p2 = points(j,:);
            y1 = p1(1);
            x1 = p1(2);
            y2 = p2(1);
            x2 = p2(2);
            d = norm(p1-p2);
            
            if (d > maxd)
%                disp(x1 + ":"+x2);
%                disp(y1 + ":"+y2);
               
               maxd = d;
               mx1 = x1;
               mx2 = x2;
               my1 = y1;
               my2 = y2;
            end
        end
    end

end


function plotPoint(x, y)
    %hold on;
    plot(x, y, 'gs', 'MarkerSize', 3);
end

function plotPoints(points)
    [n,~] = size(points);
    
    for i = 1:n
        p = points(i,:);
        y = p(1);
        x = p(2);
        plotPoint(x, y);
    end
end

function [r, p] = multipleFindMaxSupportLine(hmatrix, binRhoSize,...
    binThetaSize, maxRho, points, iters)
    %finds the top [iters] lines with max support
    %returns the inliers and a list of lines
    
    p = zeros(0, 2);
    r = zeros(iters, 4);
    
    for i = 1:iters
        [inliers, line, hmatrix] = findMaxSupportLine(hmatrix, binRhoSize, binThetaSize, maxRho, points);
        %disp(line);
        r(i, :) = line;
        p = vertcat(p, inliers);
    end
end

function [inliers, line, hmatrix] = findMaxSupportLine(hmatrix, binRhoSize,...
        binThetaSize, maxRho, points)
    %finds the line with max support in hmatrix
    %returns the inliers (points that voted for the line)
    %returns hmatrix with the point of max support removed
    %returns line (1x4 matrix of 4 points) between extremes
    [t, r] = find(ismember(hmatrix, max(hmatrix(:))));
    t = t(1);
    r = r(1);

%     disp("support: " + hmatrix(t, r) );
%     disp("t/r: "+[t, r]);
    
    hmatrix(t, r) = 0;
    
    [n,~] = size(points);
    inliers = zeros(0, 2);
    
    for i = 1:n
        p = points(i,:);
        px = p(2);
        py = p(1);
        if (pointVotedInBin(px, py, binRhoSize, binThetaSize, t, r, maxRho))
            inliers = vertcat(inliers, p);
        end
    end
    [x1, x2, y1, y2] = extremes(inliers);
    line = [x1, x2, y1, y2];
    
end

function H = hough(points, binRhoSize, binThetaSize, maxRho)
    
    %binsize is based on multiples.
    %a binsize of 1 will mean there will be a bin for each possible integer
    %a binsize of 2 will mean there will be a bin for each possible even integer
    
    [n, ~] = size(points);
    
    H = zeros(ceil(181/binThetaSize),ceil((maxRho*2 + maxRho + 1)/binRhoSize));

    for i = 1:n
        p = points(i,:);
        y = p(1);
        x = p(2);

        for t = 1:180 %0 and 180 are the same
            r = (x * cosd(t) + y * sind(t));
            r = r + maxRho + 1;
            %disp(r);
            
            rho = ceil(r/binRhoSize);
            theta = ceil((t)/binThetaSize);
            
            H(theta, rho) = H(theta, rho) + 1;
        end
    end


end

function drawLines(lines)
    %provided a matrix of lines, as in multipleRansac, draws the lines
    [n,~] = size(lines);
    
    for i = 1:n
        l = lines(i,:);
        x1 = l(1);
        x2 = l(2);
        y1 = l(3);
        y2 = l(4);
        
        plot([x1, x2],[y1, y2]);
        
        
    end
end

function [r, p] = multipleRansac(points, n, d, c, iters)
    %applies ransac <iters> times.
    %returns matrix of lines, where a line is a 1x4 matrix of 4 values
    
    r = zeros(iters, 4);
    p = zeros(0, 2); %matrix of inliers
    
    for i = 1:iters
        [x1, y1, x2, y2, points, inliers] = ransac(points, n, d, c);
        r(i, :) = [x1, x2, y1, y2];
        p = vertcat(p, inliers);
    end
    
end

function [x1, y1, x2, y2, outliers, in] = ransac(points, n, d, c)
    %applies ransac on the dataset given by points
    %returns the extreme points of a line with strong support
    %runs at most n times
    %"strong support" defined as having >= c points <= d distance from line
    
    
    for i = 1:n
        [s, ~] = size(points);
        r = randperm(s, 2);
        r1 = r(1);
        r2 = r(2);

        p1 = points(r1,:);
        y1 = p1(1);
        x1 = p1(2);

        p2 = points(r2,:);
        y2 = p2(1);
        x2 = p2(2);

        [in, outliers] = closeToLine(x1, y1, x2, y2, d, points);
        [nIn, ~] = size(in);
        if (nIn >= c)
            %must find extreme points still maybe
            [x1, x2, y1, y2] = extremes(in);
            return;
        end
    end
    %failed to find points in n iterations, return 0s.
    outliers = points;
    in = zeros(0, 2);
    x1=0;
    x2=0;
    y1=0;
    y2=0;
    disp("ERROR: Unable to find an acceptable line in " + n + " iterations.");
    disp("Consider allowing more iterations, or relax requirements for a line.");
%     disp("You can do this by increasing distance or decreasing count.");
%     disp("Decreasing threshold and sigma will also make finding a line easier.");
    
end

function [inliers, outliers] = closeToLine(x1, y1, x2, y2, d, points)
    %given a line a set of points, and a distance d, return the
    %inliers  (points <= d away from the line)
    %outliers (points > d away from the line)    
    
    inliers = points;
    outliers = points;
    blk = zeros(1,2);
    
    [n, ~] = size(points);
    
    for i = 1:n
        p = points(i,:);
        y0 = p(1);
        x0 = p(2);
        %
        if distToLine(x1, y1, x2, y2, x0, y0) <= d
            %inlier!
            outliers(i, :) = blk;
            
        else
            inliers(i, :) = blk;
        end
    end
    
    %now all points that are incorrect in each matrix is 0,0
    %remove all 0,0 rows from matrix
    inliers = inliers(~all(inliers == 0, 2),:);
    outliers = outliers(~all(outliers == 0, 2),:);
    
end

function dist = distToLine(x1, y1, x2,  y2, x0, y0)
    %returns perpendicular distance from line defined by x1 y1 x2 y2 to
    %point defined by x0 y0
    
    n = abs((x2-x1)*(y1-y0) - (x1-x0)*(y2-y1));
    d = sqrt((x2-x1)^2 + (y2-y1)^2);
    dist = n/d;
end

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
               %nms(i,j) = 1;
               points = [points;[i j]];
            end            
        end
    end
    
    nms = nms(2:m-1, 2:n-1);
    
end

function hessImg = applyHessian(vSobel, hSobel)

    Ixx = sobel(vSobel, 'v');
    Iyy = sobel(hSobel, 'h');
    Ixy = sobel(hSobel, 'v');
    
    hessImg = (Ixx.*Iyy)-(Ixy).^2;
    
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

