function varargout = gmm(X, K_or_centroids)
% ============================================================
% Expectation-Maximization iteration implementation of
% Gaussian Mixture Model.
%
% PX = GMM(X, K_OR_CENTROIDS)
% [PX MODEL] = GMM(X, K_OR_CENTROIDS)
%
%  - X: N-by-D data matrix.
%  - K_OR_CENTROIDS: either K indicating the number of
%       components or a K-by-D matrix indicating the
%       choosing of the initial K centroids.
%
%  - PX: N-by-K matrix indicating the probability of each
%       component generating each point.
%  - MODEL: a structure containing the parameters for a GMM:
%       MODEL.Miu: a K-by-D matrix.
%       MODEL.Sigma: a D-by-D-by-K matrix.
%       MODEL.Pi: a 1-by-K vector.
%
%
%  inv 矩阵的求逆命令；
%  repmat 通过扩展向量到矩阵
%  min 求矩阵最小值，可以返回标签 
%  X(labels == k, : ) 对行做筛选 
%  size(Xk, 1) 求矩阵的长或宽
%  单位向量矩阵 I=diag(diag(ones(D)))  I is a D-by-D matrix
%  sum(matrix,1) 对matrix的每一列求和，若为2，则是对每一行求和
%  diag函数功能:矩阵对角元素的提取和创建对角阵
%  scatter 对二维向量绘图
% ============================================================

threshold = 1e-15;
[N, D] = size(X);
% isscalar 判断是否为标量
if isscalar(K_or_centroids)
    K = K_or_centroids;
    % randomly pick centroids
    rndp = randperm(N);
    centroids = X(rndp(1:K), :);       %初始化中心
else  % 矩阵，给出每一类的初始化
    K = size(K_or_centroids, 1);
    centroids = K_or_centroids;
end

% initial values
[pMiu pPi pSigma] = init_params();

Lprev = -inf;
% 计算和更新pPi,pMiu,pSigma
while true
    %% Estiamtion Step
    Px = calc_prob();         %% Px表示由K个model产生时每个model贡献的概率，a N-by-D matrix

    % new value for pGamma
    pGamma = Px .* repmat(pPi, N, 1);             %估计 gamma 是个N*K的矩阵,见公式（7）
    pGamma = pGamma ./ repmat(sum(pGamma, 2), 1, K);     %每个样本又第k类聚类，又称componen的生成概率

    %% Maximization Step
    % new value for parameters of each Component
    Nk = sum(pGamma, 1);               %diag(1./Nk)将1./Nk变为对角矩阵
    pMiu = diag(1./Nk) * pGamma' * X;  %更新pMiu
    pPi = Nk/N;                        %更新pPi
    for kk = 1:K
        Xshift = X-repmat(pMiu(kk, :), N, 1);
        pSigma(:, :, kk) = (Xshift' * (diag(pGamma(:, kk)) * Xshift)) / Nk(kk);     %更新sigma
    end

    %% check for convergence
    %不断迭代EM步骤，更新参数，直到似然函数前后差值小于一个阈值，或者参数前后之间的差（一般选择欧式距离）小于某个阈值，终止迭代，这里选择前者
    L = sum(log(Px*pPi'));   % pPi is a 1-by-K matrix
    if L-Lprev < threshold
        break;
    end
    Lprev = L;
end

% 输出参数判定
if nargout == 1
    varargout = {Px};
else
    model = [];
    model.Miu = pMiu;
    model.Sigma = pSigma;
    model.Pi = pPi;
    varargout = {Px, model};
    %varargout = {pGamma, model};%注意！！！！！这里和大神代码不同，他返回的是px，而我是 pGamma
end

function [pMiu pPi pSigma] = init_params()
    pMiu = centroids;  % 均值，也就是K类的中心
    pPi = zeros(1, K); % 概率
    pSigma = zeros(D, D, K); %协方差矩阵，每个都是 D*D

    % hard assign x to each centroids   %repmat 为复制和平铺矩阵  % X is a N-by-D data matrix.
    % (X - pMiu)^2 = X^2 + pMiu^2 - 2*X*pMiu                     % X->K列 U->N行 XU^T is N-by-K 
    distmat = repmat(sum(X.*X, 2), 1, K) + repmat(sum(pMiu.*pMiu, 2)', N, 1) - 2*X*pMiu';  %计算每个点到K个中心的距离
    [dummy labels] = min(distmat, [], 2);    %找到离X最近的pMiu，[C,I] labels代表这个最小值是从那列选出来的，dummy为每个点到最近的K的距离

    for k=1:K   %初始化参数
        Xk = X(labels == k, :);           %Xk是所有被归到K类的X向量构成的矩阵,Xk为第K个模型中的点的信息，labels==K 表示第k个模型出现的位置，
        pPi(k) = size(Xk, 1)/N;           %数一数几个归到K类的,计算第K个模型出现的概率
        pSigma(:, :, k) = cov(Xk);        %计算Xk的协方差矩阵 D-by-D matrix,最小方差无偏估计
    end
end

% 计算概率，见公式（1）
function Px = calc_prob()
    Px = zeros(N, K);
    for k = 1:K
        Xshift = X-repmat(pMiu(k, :), N, 1);   % (x-u)
        inv_pSigma = inv(pSigma(:, :, k)+diag(repmat(threshold,1,size(pSigma(:, :, k),1)))); % 方差矩阵求逆,考虑了奇异值矩阵问题
        tmp = sum((Xshift*inv_pSigma) .* Xshift, 2);
        coef = (2*pi)^(-D/2) * sqrt(det(inv_pSigma)); % det 求方差矩阵的行列式  
        Px(:, k) = coef * exp(-0.5*tmp);
    end
end

% 计算概率,另外一种考虑协方差矩阵如果是奇异值矩阵情况
function Px = calc_prob1()
        Px = zeros(N,K);
        for k = 1:K
            Xshift = X-repmat(pMiu(k, :), N, 1);   % (x-u)
            lemda=1e-5;
            conv = pSigma(:, :, k) + lemda * diag(diag(ones(D)));      %%直接用pSigma(:, :, k)，可能会是奇异矩阵，行列式值为0，不能计算概率，这里处理singular问题，为协方差矩阵加上一个很小的lemda*I
            inv_pSigma = inv(conv);%协方差的逆
            tmp = sum((Xshift*inv_pSigma) .* Xshift, 2);%(X-U_k)sigma.*(X-U_k),tmp是个N*1的向量
            coef = (2*pi)^(-D/2) * sqrt(det(inv_pSigma)); % det 求方差矩阵的行列式 
            Px(:, k) = coef * exp(-0.5*tmp);  %把数据点 x 带入到 Gaussian model 里得到的值 ,依据公式（1）高斯的概率公式
        end
end
end

