clc;
%可修改参数
%LenLeft  提取长度 （左LenLeft个字符）
%LenRight  提取长度 （右LenRight个字符）
%xlsFileName   存放正位点位置xls表格，注意数据必须在第一个工作表中
%fastaFileName  存放下载序列fasta文件
%执行完文件会放入pos.txt和neg.txt中，重新执行需将二文件删除，否则新数据追加到旧数据后。
LenLeft=10;
LenRight=10;
%xlsFileName='Hydroxyproline.xls'; 
xlsFileName='citrullination-ID.xlsx'; 
[posnum,txtName]=xlsread(xlsFileName);
fastaFileName='citrullination120.fasta';
 [row,col]=size(txtName);
 txtName=txtName(2:row,1);
 row=row-1;
 fidin=fopen(fastaFileName,'r');
 newline=0;
 newName=[];
 tmpSeq=[];
 while ~feof(fidin)
     tline=fgetl(fidin);
     if (tline(1)=='>')
           k=5;
          while  tline(k)~='|'
              k=k+1;
          end
          if(newline>0)
              disp([newName,' ',tmpSeq]);
               position=[];
              for j=1:row
                  
                  if( strcmp(newName,txtName(j))==1)
                          %disp([newName,' VS ',txtName(j)]);   
                          position=[position,posnum(j)];
                  end
              end   
              disp([newName, num2str(position)]);
              genPosNegtxtLR(newName,tmpSeq,position,LenLeft,LenRight);
          end
          newName=tline(5:(k-1));
          newline=newline+1;
          tmpSeq=[];
      else
         tmpSeq=[tmpSeq,tline];
     end
       
 end
 fclose(fidin);
 %disp(tmpSeq);
if(newline>0)
           disp([newName,' ',tmpSeq]);
           position=[];
           for j=1:row
                  if( strcmp(newName,txtName(j))==1)
                          %disp([newName,' VS ',txtName(j)]);   
                           position=[position,posnum(j)];
                  end
              end   
              disp([newName, num2str(position)]);
              genPosNegtxtLR(newName,tmpSeq,position,LenLeft,LenRight);
end
          