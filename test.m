
load fisheriris.mat;
%Use on our class example
Data = iris_dataset';
Labels = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3];
[N,Dim] = size(Data);
%Polynomial
H = zeros(length(Data));
len = length(Data);
for i=1:len
    for j=1:len
        K=(Data(j,:)*Data(i,:)' + 1)^2;
        H(i,j) = Labels(i)*Labels(j)*K;
    end
end
ff = -ones(N,1);
Aeq = zeros(N,N);
Aeq(1,:) = Labels;
beq = zeros(N,1);
Cval = 10;
[x,fval,exitflag,output,lambda] = quadprog(H+eye(N)*0.0001,ff,[],[],Aeq,beq,zeros(1,N),Cval*ones(1,N));
supportVectors = find(x > eps);
supportX = x(supportVectors);
supportData = Data(supportVectors,:);
supportLabels = Labels(supportVectors);
supportLength = length(supportLabels);
%Now, solve for b
%Create a set of b's and average them
Bset = [];
for i=1:supportLength
Bval = 0;
for j=1:supportLength
K=(supportData(i,:)*supportData(j,:)' + 1)^2;
Bval = Bval + ( supportX(j) * supportLabels(j) * K );
end
Bval = supportLabels(i) * Bval;
Bval = (1 - Bval)/supportLabels(i);
Bset = [ Bset Bval ];
end
b = mean(Bset);
Res = zeros(1,N);
for i=1:N
sumVal = 0;
for j=1:supportLength
K=(supportData(j,:)*Data(i,:)' + 1)^2;
sumVal = sumVal + supportX(j)*supportLabels(j)*K;
end
Res(i) = sumVal + b;
end
