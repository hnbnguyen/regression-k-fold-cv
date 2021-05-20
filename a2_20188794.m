function [rmsvars lowndx rmstrain rmstest] = a2_20188794
% [RMSVARS LOWNDX RMSTRAIN RMSTEST]=A3 finds the RMS errors of
% linear regression of the data in the file "AIRQUALITY.CSV" by treating
% each column as a vector of dependent observations, using the other
% columns of the data as observations of independent varaibles. The
% individual RMS errors are returned in RMSVARS and the index of the
% smallest RMS error is returned in LOWNDX. For the variable that is
% best explained by the other variables, a 5-fold cross validation is
% computed. The RMS errors for the training of each fold are returned
% in RMSTEST and the RMS errors for the testing of each fold are
% returned in RMSTEST.
%
% INPUTS:
%         none
% OUTPUTS:
%         RMSVARS  - 1xN array of RMS errors of linear regression
%         LOWNDX   - integer scalar, index into RMSVALS
%         RMSTRAIN - 1x5 array of RMS errors for 5-fold training
%         RMSTEST  - 1x5 array of RMS errors for 5-fold testing

    filename = 'airquality.csv';
    [rmsvars lowndx] = a2q1(filename);
    [rmstrain rmstest] = a2q2(filename, lowndx);
    
    

end

function [rmsvars lowndx] = a2q1(filename)
% [RMSVARS LOWNDX]=A2Q1(FILENAME) finds the RMS errors of
% linear regression of the data in the file FILENAME by treating
% each column as a vector of dependent observations, using the other
% columns of the data as observations of independent varaibles. The
% individual RMS errors are returned in RMSVARS and the index of the
% smallest RMS error is returned in LOWNDX. 
%
% INPUTS:
%         FILENAME - character string, name of file to be processed;
%                    assume that the first row describes the data variables
% OUTPUTS:
%         RMSVARS  - 1xN array of RMS errors of linear regression
%         LOWNDX   - integer scalar, index into RMSVALS

    % Read the test data from a CSV file; find the size of the data
    % %
    Amat = csvread(filename, 1, 0);
    Amat(Amat<0) = nan;
    initial_shape = size(Amat)
    threshold = round(size(Amat, 2) * 2/3);
    
    Amat(sum(isnan(Amat), 2) >= threshold, :) = [];
    post_preprocessing_shape = size(Amat)
    Amat = fillmissing(Amat, 'linear');
    n_Amat = normalize(Amat);
    

    % Compute the RMS errors for linear regression
    % %
    % %
    [rownum, colnum] = size(Amat);
    rmsvars = zeros(1, colnum);
    
    for idx = 1 : colnum
        cvec = n_Amat(:,idx);
        Xmat = n_Amat;
        Xmat(:, idx) = [];
        % Adding an intercept term to the model
        Xmat = [Xmat ones(length(Xmat),1)];
        uval = linsolve(Xmat,cvec);
        rmsvars(idx) = rms(cvec - Xmat*uval);
    end 
    
    rmsvars
    lowndx = find(rmsvars == min(rmsvars))

    % Find the regression in unstandardized variables
    % %
    % % Assuming it's doing regression unnormalized data
    unstandardized_rmsvars = zeros(1, colnum);
    for jdx = 1 : colnum
        u_cvec = Amat(:,jdx);
        u_Xmat = Amat;
        u_Xmat(:, jdx) = [];
        u_Xmat = [u_Xmat ones(length(u_Xmat), 1)];
        u = linsolve(u_Xmat, u_cvec);
        unstandardized_rmsvars(jdx) = rms(u_cvec - u_Xmat*u);
    end
    unstandardized_rmsvars
    % 
    
    % Plot the results of the lowndx column
    % %
    v_lowndx = n_Amat(:, lowndx);
    t = 1:numel(v_lowndx);
    plot(t(100:100:end),v_lowndx(100:100:end), "*-");
    title("Linear Regression of Best-Modeled Chemical");
    ylabel("Chemical Concentration");
    xlabel("Time")
    
                            
end
function [rmstrain rmstest] = a2q2(filename,lowndx)
% [RMSTRAIN RMSTEST]=A3Q2(LOWNDX) finds the RMS errors of 5-fold
% cross-validation for the variable LOWNDX of the data in the file
% FILENAME. The RMS errors for the training of each fold are returned
% in RMSTEST and the RMS errors for the testing of each fold are
% returned in RMSTEST.
%
% INPUTS:
%         FILENAME - character string, name of file to be processed;
%                    assume that the first row describes the data variables
%         LOWNDX   - integer scalar, index into the data
% OUTPUTS:
%         RMSTRAIN - 1x5 array of RMS errors for 5-fold training
%         RMSTEST  - 1x5 array of RMS errors for 5-fold testing

    % Read the test data from a CSV file; find the size of the data
    % %
    % % STUDENT CODE GOES HERE: REMOVE THIS COMMENT
    % %
    Amat = csvread(filename, 1, 0);
    Amat(Amat<0) = nan;
    
    threshold = round(size(Amat, 2) * 2/3);
    
    Amat(sum(isnan(Amat), 2) >= threshold, :) = [];
    Amat = fillmissing(Amat, 'linear');
    % Create Xmat and yvec from the data and the input parameter,
    % accounting for no standardization of data
    % %
    Xmat = Amat;
    Xmat(:,lowndx) = [];
    yvec = Amat(:, lowndx);

    % Compute the RMS errors of 5-fold cross-validation
    % %
    % % STUDENT CODE GOES HERE: REMOVE THE NEXT 2 LINES AND THIS COMMENT
    % %
    [rmstrain rmstest] = mykfold(Xmat, yvec, 5);

end

function [rmstrain,rmstest]=mykfold(Xmat, yvec, k_in)
% [RMSTRAIN,RMSTEST]=MYKFOLD(XMAT,yvec,K) performs a k-fold validation
% of the least-squares linear fit of yvec to XMAT. If K is omitted,
% the default is 5.
%
% INPUTS:
%         XMAT     - MxN data vector
%         yvec     - Mx1 data vector
%         K        - positive integer, number of folds to use
% OUTPUTS:
%         RMSTRAIN - 1xK vector of RMS error of the training fits
%         RMSTEST  - 1xK vector of RMS error of the testing fits

    % Problem size
    M = size(Xmat, 1); % gettting the number of rows  - samples
    
    

    % Set the number of folds; must be 1<k<M
    if nargin >= 3 & ~isempty(k_in)
        k = max(min(round(k_in), M-1), 2);
    else
        k = 5;
    end

    % Initialize the return variables
    rmstrain = zeros(1, k);
    rmstest  = zeros(1, k);
    
    %creating my own crossval algorithm
    %generate a vector size M of random indices
    random_number = randperm(M)';
    indices = zeros(M, 1);
    
    for cdx = 1: M
        indices(cdx) = mod(random_number(cdx),k) + 1;
    end
    
    for ix=1:k
        % %
        % % STUDENT CODE GOES HERE: replace the next 5 lines with code to
        % % (1) set up the "train" and "test" indexing for "xmat" and "yvec"
        % % (2) use the indexing to set up the "train" and "test" data
        % % (3) compute "wvec" for the training data
        % %
        
        test = (indices == ix);
        train = ~test;
%                 
        xmat_train = Xmat(train,:);
        xmat_train = [xmat_train ones(length(xmat_train), 1)];
        xmat_test = Xmat(test,:);
        xmat_test = [xmat_test ones(length(xmat_test), 1)];
        
        yvec_train = yvec(train,:);
        yvec_test = yvec(test,:);
        
        wvec = linsolve(xmat_train,yvec_train);

        rmstrain(ix) = rms(xmat_train*wvec - yvec_train);
        rmstest(ix)  = rms(xmat_test*wvec  - yvec_test);
    end
    rmstrain
    rmstest
    var_train = std(rmstrain);
    var_test = std(rmstest);

end
