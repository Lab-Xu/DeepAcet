function  ErrorMessage=genDataTxt(newName,tmpSeq,position,Len)
if(length(position)>0)
  ModSeq=tmpSeq(position(1));
  for i=1:Len
    ModSeq=['X',ModSeq,'X'];
  end   
  fidout=fopen('data.txt','a');
  for i=1:length(position)
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
end 
   
 







