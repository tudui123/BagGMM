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
%  inv ������������
%  repmat ͨ����չ����������
%  min �������Сֵ�����Է��ر�ǩ 
%  X(labels == k, : ) ������ɸѡ 
%  size(Xk, 1) �����ĳ����
%  ��λ�������� I=diag(diag(ones(D)))  I is a D-by-D matrix
%  sum(matrix,1) ��matrix��ÿһ����ͣ���Ϊ2�����Ƕ�ÿһ�����
%  diag��������:����Խ�Ԫ�ص���ȡ�ʹ����Խ���
%  scatter �Զ�ά������ͼ
% ============================================================

threshold = 1e-15;
[N, D] = size(X);
% isscalar �ж��Ƿ�Ϊ����
if isscalar(K_or_centroids)
    K = K_or_centroids;
    % randomly pick centroids
    rndp = randperm(N);
    centroids = X(rndp(1:K), :);       %��ʼ������
else  % ���󣬸���ÿһ��ĳ�ʼ��
    K = size(K_or_centroids, 1);
    centroids = K_or_centroids;
end

% initial values
[pMiu pPi pSigma] = init_params();

Lprev = -inf;
% ����͸���pPi,pMiu,pSigma
while true
    %% Estiamtion Step
    Px = calc_prob();         %% Px��ʾ��K��model����ʱÿ��model���׵ĸ��ʣ�a N-by-D matrix

    % new value for pGamma
    pGamma = Px .* repmat(pPi, N, 1);             %���� gamma �Ǹ�N*K�ľ���,����ʽ��7��
    pGamma = pGamma ./ repmat(sum(pGamma, 2), 1, K);     %ÿ�������ֵ�k����࣬�ֳ�componen�����ɸ���

    %% Maximization Step
    % new value for parameters of each Component
    Nk = sum(pGamma, 1);               %diag(1./Nk)��1./Nk��Ϊ�ԽǾ���
    pMiu = diag(1./Nk) * pGamma' * X;  %����pMiu
    pPi = Nk/N;                        %����pPi
    for kk = 1:K
        Xshift = X-repmat(pMiu(kk, :), N, 1);
        pSigma(:, :, kk) = (Xshift' * (diag(pGamma(:, kk)) * Xshift)) / Nk(kk);     %����sigma
    end

    %% check for convergence
    %���ϵ���EM���裬���²�����ֱ����Ȼ����ǰ���ֵС��һ����ֵ�����߲���ǰ��֮��Ĳһ��ѡ��ŷʽ���룩С��ĳ����ֵ����ֹ����������ѡ��ǰ��
    L = sum(log(Px*pPi'));   % pPi is a 1-by-K matrix
    if L-Lprev < threshold
        break;
    end
    Lprev = L;
end

% ��������ж�
if nargout == 1
    varargout = {Px};
else
    model = [];
    model.Miu = pMiu;
    model.Sigma = pSigma;
    model.Pi = pPi;
    varargout = {Px, model};
    %varargout = {pGamma, model};%ע�⣡������������ʹ�����벻ͬ�������ص���px�������� pGamma
end

function [pMiu pPi pSigma] = init_params()
    pMiu = centroids;  % ��ֵ��Ҳ����K�������
    pPi = zeros(1, K); % ����
    pSigma = zeros(D, D, K); %Э�������ÿ������ D*D

    % hard assign x to each centroids   %repmat Ϊ���ƺ�ƽ�̾���  % X is a N-by-D data matrix.
    % (X - pMiu)^2 = X^2 + pMiu^2 - 2*X*pMiu                     % X->K�� U->N�� XU^T is N-by-K 
    distmat = repmat(sum(X.*X, 2), 1, K) + repmat(sum(pMiu.*pMiu, 2)', N, 1) - 2*X*pMiu';  %����ÿ���㵽K�����ĵľ���
    [dummy labels] = min(distmat, [], 2);    %�ҵ���X�����pMiu��[C,I] labels���������Сֵ�Ǵ�����ѡ�����ģ�dummyΪÿ���㵽�����K�ľ���

    for k=1:K   %��ʼ������
        Xk = X(labels == k, :);           %Xk�����б��鵽K���X�������ɵľ���,XkΪ��K��ģ���еĵ����Ϣ��labels==K ��ʾ��k��ģ�ͳ��ֵ�λ�ã�
        pPi(k) = size(Xk, 1)/N;           %��һ�������鵽K���,�����K��ģ�ͳ��ֵĸ���
        pSigma(:, :, k) = cov(Xk);        %����Xk��Э������� D-by-D matrix,��С������ƫ����
    end
end

% ������ʣ�����ʽ��1��
function Px = calc_prob()
    Px = zeros(N, K);
    for k = 1:K
        Xshift = X-repmat(pMiu(k, :), N, 1);   % (x-u)
        inv_pSigma = inv(pSigma(:, :, k)+diag(repmat(threshold,1,size(pSigma(:, :, k),1)))); % �����������,����������ֵ��������
        tmp = sum((Xshift*inv_pSigma) .* Xshift, 2);
        coef = (2*pi)^(-D/2) * sqrt(det(inv_pSigma)); % det �󷽲���������ʽ  
        Px(:, k) = coef * exp(-0.5*tmp);
    end
end

% �������,����һ�ֿ���Э����������������ֵ�������
function Px = calc_prob1()
        Px = zeros(N,K);
        for k = 1:K
            Xshift = X-repmat(pMiu(k, :), N, 1);   % (x-u)
            lemda=1e-5;
            conv = pSigma(:, :, k) + lemda * diag(diag(ones(D)));      %%ֱ����pSigma(:, :, k)�����ܻ��������������ʽֵΪ0�����ܼ�����ʣ����ﴦ��singular���⣬ΪЭ����������һ����С��lemda*I
            inv_pSigma = inv(conv);%Э�������
            tmp = sum((Xshift*inv_pSigma) .* Xshift, 2);%(X-U_k)sigma.*(X-U_k),tmp�Ǹ�N*1������
            coef = (2*pi)^(-D/2) * sqrt(det(inv_pSigma)); % det �󷽲���������ʽ 
            Px(:, k) = coef * exp(-0.5*tmp);  %�����ݵ� x ���뵽 Gaussian model ��õ���ֵ ,���ݹ�ʽ��1����˹�ĸ��ʹ�ʽ
        end
end
end

