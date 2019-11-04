clear;
clear all;
close all;
clc;
nz=400;
nx=800;
dx=10;dz=10;
xx=0:dx:dx*(nx-1);
zz=0:dz:dz*(nz-1);
xx=xx/1000;
zz=zz/1000;

%% reconstruction
fid=fopen('T2_3_5600vx0.dat','rb');  %T2 0.6
fid2=fopen('T2_6_2800vx0.dat','rb');    %T2 1.2
fid3=fopen('T2_12_1400vx0.dat','rb');     %T4 1.4
fid4=fopen('T4_14_1200vx0.dat','rb');   
fid5=fopen('T4_16_1050vx0.dat','rb'); 

p203=fread(fid,[nz,nx],'float');
p206=fread(fid2,[nz,nx],'float');
p212=fread(fid3,[nz,nx],'float');
p414=fread(fid4,[nz,nx],'float');
p416=fread(fid5,[nz,nx],'float');


maxvalue=(max(max(p203)));
minvalue=(max(max(p203)));
if maxvalue<abs(minvalue)
    maxvalue=abs(minvalue)
end
p203=p203/maxvalue;

maxvalue=(max(max(p206)));
minvalue=(max(max(p206)));
if maxvalue<abs(minvalue)
    maxvalue=abs(minvalue)
end
p206=p206/maxvalue;

% 
maxvalue=(max(max(p212)));
minvalue=(max(max(p212)));
if maxvalue<abs(minvalue)
    maxvalue=abs(minvalue);
end
p212=p212/maxvalue;

maxvalue=(max(max(p414)));
minvalue=(max(max(p414)));
if maxvalue<abs(minvalue)
    maxvalue=abs(minvalue);
end
p414=p414/maxvalue;
% 
maxvalue=(max(max(p416)));
minvalue=(max(max(p416)));
if maxvalue<abs(minvalue)
    maxvalue=abs(minvalue);
end
p416=p416/maxvalue;


fclose(fid);fclose(fid2);
fclose(fid3);fclose(fid4);fclose(fid5);


figure(1)
imagesc(xx,zz,p203);colormap(gray);mx=max(max(p203));mn=min(min(p203));caxis([mn mx]/30);set(gca,'xaxislocation','top'); xlabel('x /(km)');ylabel('z /(km)');colorbar;%title('vz seismogram ');
set(gcf,'Position',[100 100 280 160]); colorbar;
set(gca,'Position',[.13 .08 .680 .65]); colorbar('position',[0.83 0.08 0.04 0.6]);
set(gca,'xaxislocation','top'); xlabel('X (km)');ylabel('Z (km)');
figure_FontSize=18;   
set(get(gca,'XLabel'),'FontSize',figure_FontSize); 
set(get(gca,'YLabel'),'FontSize',figure_FontSize); 
set(gca,'FontSize',figure_FontSize);  
set(findobj(get(gca,'Children'),'LineWidth',0.5),'LineWidth',2); 
set(gcf, 'PaperPositionMode', 'manual');
caxis([-0.04 0.03]);

print (gcf,'-deps','-r300','./p203.eps');

figure(2)
imagesc(xx,zz,p206);colormap(gray);mx=max(max(p206));mn=min(min(p206));caxis([mn mx]/30);set(gca,'xaxislocation','top'); xlabel('x /(km)');ylabel('z /(km)');colorbar;%title('vx seismogram');
set(gcf,'Position',[100 100 280 160]); colorbar;
set(gca,'Position',[.13 .08 .680 .65]); colorbar('position',[0.83 0.08 0.04 0.6]);
set(gca,'xaxislocation','top'); xlabel('X (km)');ylabel('Z (km)');
figure_FontSize=18;   
set(get(gca,'XLabel'),'FontSize',figure_FontSize); 
set(get(gca,'YLabel'),'FontSize',figure_FontSize); 
set(gca,'FontSize',figure_FontSize);  
set(findobj(get(gca,'Children'),'LineWidth',0.5),'LineWidth',2); 
set(gcf, 'PaperPositionMode', 'manual');
caxis([-0.04 0.03]);

print (gcf,'-deps','-r300','./p206.eps');

figure(3)
imagesc(xx,zz,p212);colormap(gray);mx=max(max(p212));mn=min(min(p212));caxis([mn mx]/30);set(gca,'xaxislocation','top'); xlabel('x /(km)');ylabel('z /(km)');colorbar;%title('vz seismogram ');
set(gcf,'Position',[100 100 280 160]); colorbar;
set(gca,'Position',[.13 .08 .680 .65]); colorbar('position',[0.83 0.08 0.04 0.6]);
set(gca,'xaxislocation','top'); xlabel('X (km)');ylabel('Z (km)');
figure_FontSize=18;   
set(get(gca,'XLabel'),'FontSize',figure_FontSize); 
set(get(gca,'YLabel'),'FontSize',figure_FontSize); 
set(gca,'FontSize',figure_FontSize);  
set(findobj(get(gca,'Children'),'LineWidth',0.5),'LineWidth',2); 
set(gcf, 'PaperPositionMode', 'manual');
caxis([-0.04 0.03]);

print (gcf,'-deps','-r300','./p212.eps');

figure(4)
imagesc(xx,zz,p414);colormap(gray);mx=max(max(p414));mn=min(min(p414));caxis([mn mx]/30);set(gca,'xaxislocation','top'); xlabel('x /(km)');ylabel('z /(km)');colorbar;%title('vx seismogram');
set(gcf,'Position',[100 100 280 160]); colorbar;
set(gca,'Position',[.13 .08 .680 .65]); colorbar('position',[0.83 0.08 0.04 0.6]);
set(gca,'xaxislocation','top'); xlabel('X (km)');ylabel('Z (km)');
figure_FontSize=18;   
set(get(gca,'XLabel'),'FontSize',figure_FontSize); 
set(get(gca,'YLabel'),'FontSize',figure_FontSize); 
set(gca,'FontSize',figure_FontSize);  
set(findobj(get(gca,'Children'),'LineWidth',0.5),'LineWidth',2); 
set(gcf, 'PaperPositionMode', 'manual');
caxis([-0.04 0.03]);

print (gcf,'-deps','-r300','./p414.eps');

figure(5)
imagesc(xx,zz,p416);colormap(gray);mx=max(max(p416));mn=min(min(p416));caxis([mn mx]/30);set(gca,'xaxislocation','top'); xlabel('x /(km)');ylabel('z /(km)');colorbar;%title('vz seismogram ');
set(gcf,'Position',[100 100 280 160]); colorbar;
set(gca,'Position',[.13 .08 .680 .65]); colorbar('position',[0.83 0.08 0.04 0.6]);
set(gca,'xaxislocation','top'); xlabel('X (km)');ylabel('Z (km)');
figure_FontSize=18;   
set(get(gca,'XLabel'),'FontSize',figure_FontSize); 
set(get(gca,'YLabel'),'FontSize',figure_FontSize); 
set(gca,'FontSize',figure_FontSize);  
set(findobj(get(gca,'Children'),'LineWidth',0.5),'LineWidth',2); 
set(gcf, 'PaperPositionMode', 'manual');
caxis([-0.04 0.03]);

print (gcf,'-deps','-r300','./p416.eps');

line203=p203(350,:);
line206=p206(350,:);
line212=p212(350,:);
line414=p414(350,:);
line416=p416(350,:);


figure(6)
set(gcf,'Position',[100 100 600 240]); 
set(gcf,'Position',[100 300 600 240]);
axis();
h=plot(xx,line203,'-xg',xx,line206,'-b',xx,line212,'-.r',xx, line414,'-+k',xx, line416,'-*c','markersize',2);
xlabel('X (km)','fontsize',8);
ylabel('Amplitude','fontsize',8);
set(gca,'fontsize',8);
axis([5.5 8 -0.7 0.55]);
set(h,'LineWidth',0.1); 
legend(h,'T2 0.3 ms','T2 0.6 ms','T2 1.2 ms','T4 1.4 ms','T4 1.6 ms','Fontsize',6);

print (gcf,'-deps','-r600','./tracex.eps');

lline203=p203(:,700);
lline206=p206(:,700);
lline212=p212(:,700);
lline414=p414(:,700);
lline416=p416(:,700);

figure(7)
set(gcf,'Position',[100 100 600 240]); 
set(gcf,'Position',[100 300 600 250]);
axis();
h=plot(zz,lline203,'-xg',zz,lline206,'-b',zz,lline212,'-.r',zz, lline414,'-+k',zz, lline416,'-*c','markersize',2);
xlabel('Z (km)','fontsize',8);
ylabel('Amplitude','fontsize',8);
set(gca,'fontsize',12);
axis([0.4 3.8 -0.46 0.4]);

set(h,'LineWidth',0.1); 
legend(h,'T2 0.3 ms','T2 0.6 ms','T2 1.2 ms','T4 1.4 ms','T4 1.6 ms','Fontsize',6);

print (gcf,'-deps','-r600','./tracez.eps');