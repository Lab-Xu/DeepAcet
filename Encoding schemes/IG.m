function IG_code=IG(seq,K)%K为窗体大小
[m1,n1]=size(seq);
amino =['A'    'C'    'D'    'E'    'F'    'G'    'H'  'I'    'K'  'L'    'M'    'N' ...
    'P'    'Q'   'R'    'S'    'T'    'V'    'W'    'Y'   'X' ];%氨基酸
IGE=[];%用于存储seq的信息增益值
for i=1:m1%对seq进行行循环
     H=0;%计算一行序列的信息熵
     RE=0;%计算Kullback-Leibler distance
    for k=1:21
        posi=find( seq(i,:)==amino(k));%找到每行中的氨基酸对应amnio矩阵中的氨基酸的位置
        len=length(posi);%每行中每种氨基酸的个数
        p=len/K;%计算每种氨计算占窗体长度的比例
        if p==0
            H=H;
            RE=RE;
        else
        H=H-p*log2(p);%计算一行序列的信息熵
        RE=RE+p*log2(p*K);%计算Kullback-Leibler distance
        end
        ig=H-RE;%计算每一行的信息增益
    end
    IGE=[IGE;ig]; %用于存储seq所有行的信息增益值
end
    IG_code= IGE;
end