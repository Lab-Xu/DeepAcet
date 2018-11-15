function  ErrorMessage=genPosNegtxt(newName,tmpSeq,position,Len)
keyChar=tmpSeq(position(1));
Neg=find(tmpSeq==keyChar);
ModSeq=keyChar;
for i=1:Len
    ModSeq=['X',ModSeq,'X'];
end   
disp(position);
fidout=fopen('pos.txt','a');
for i=1:length(position)
    p=find(Neg~=position(i));
    Neg=Neg(p);
    outputSeq=ModSeq;
    
    for j=1:Len
        if(position(i)==j)
            break;
        end
         outputSeq(Len+1-j)=tmpSeq(position(i)-j);
     end   
    for j=1:Len
        if(position(i)+j>length(tmpSeq))
            break;
        end
         outputSeq(Len+1+j)=tmpSeq(position(i)+j);
    end   
    fprintf( fidout,'%s %d  %s\r\n ',newName,position(i),outputSeq); 
end  
   fclose( fidout);
   
 disp(Neg);
 fidout=fopen('neg.txt','a');
   for i=1:length(Neg)
       outputSeq=ModSeq;
    
    for j=1:Len
        if(Neg(i)==j)
            break;
        end
         outputSeq(Len+1-j)=tmpSeq(Neg(i)-j);
     end   
    for j=1:Len
        if(Neg(i)+j>length(tmpSeq))
            break;
        end
         outputSeq(Len+1+j)=tmpSeq(Neg(i)+j);
    end   
    fprintf( fidout,'%s %d  %s\r\n ',newName,Neg(i),outputSeq); 
end  
   fclose( fidout);







