function IG_code=IG(seq,K)%KΪ�����С
[m1,n1]=size(seq);
amino =['A'    'C'    'D'    'E'    'F'    'G'    'H'  'I'    'K'  'L'    'M'    'N' ...
    'P'    'Q'   'R'    'S'    'T'    'V'    'W'    'Y'   'X' ];%������
IGE=[];%���ڴ洢seq����Ϣ����ֵ
for i=1:m1%��seq������ѭ��
     H=0;%����һ�����е���Ϣ��
     RE=0;%����Kullback-Leibler distance
    for k=1:21
        posi=find( seq(i,:)==amino(k));%�ҵ�ÿ���еİ������Ӧamnio�����еİ������λ��
        len=length(posi);%ÿ����ÿ�ְ�����ĸ���
        p=len/K;%����ÿ�ְ�����ռ���峤�ȵı���
        if p==0
            H=H;
            RE=RE;
        else
        H=H-p*log2(p);%����һ�����е���Ϣ��
        RE=RE+p*log2(p*K);%����Kullback-Leibler distance
        end
        ig=H-RE;%����ÿһ�е���Ϣ����
    end
    IGE=[IGE;ig]; %���ڴ洢seq�����е���Ϣ����ֵ
end
    IG_code= IGE;
end