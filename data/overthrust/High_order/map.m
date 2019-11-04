clear;
clear all;
clc;
nz=170;
nx=800;
nt=3000;
dx=12.5;dz=12.5;dt=0.0013;
xx=0:dx:dx*(nx-1);
zz=0:dz:dz*(nz-1);
xx=xx/1000;
zz=zz/1000;
tt=0:dt:dt*(nt-1);
% %% reconstruction
% fid=fopen('800vx5.dat','rb');
% fid2=fopen('800vx_inv5.dat','rb');                                                                                                                   
% p1=fread(fid,[nz,nx],'float');
% p2=fread(fid2,[nz,nx],'float');
% dp=p1-p2;
% fclose(fid);fclose(fid2);
% figure(1)
% imagesc(xx,zz,p1);colormap(gray);mx=max(max(p1));mn=min(min(p1));caxis([mn mx]/10);set(gca,'xaxislocation','top'); xlabel('x (km)');ylabel('z (km)');colorbar;%title('vx seismogram');
% set(gcf,'Position',[100 100 260 130]); colorbar;
% set(gca,'Position',[.13 .08 .680 .65]); colorbar('position',[0.83 0.08 0.04 0.6]);
% set(gca,'xaxislocation','top'); xlabel('x (km)');ylabel('z (km)');
% figure_FontSize=8;  
% set(get(gca,'XLabel'),'FontSize',figure_FontSize,'Vertical','middle'); 
% set(get(gca,'YLabel'),'FontSize',figure_FontSize,'Vertical','middle'); 
% set(findobj('FontSize',10),'FontSize',figure_FontSize);  
% set(findobj(get(gca,'Children'),'LineWidth',0.5),'LineWidth',2); 
% set(gcf, 'PaperPositionMode', 'manual');
% figure(2)
% imagesc(xx,zz,p2);colormap(gray);mx=max(max(p2));mn=min(min(p2));caxis([mn mx]/10);set(gca,'xaxislocation','top'); xlabel('x /(km)');ylabel('z /(km)');colorbar;%title('vz seismogram ');
% set(gcf,'Position',[100 100 260 130]); colorbar;
% set(gca,'Position',[.13 .08 .680 .65]); colorbar('position',[0.83 0.08 0.04 0.6]);
% set(gca,'xaxislocation','top'); xlabel('x (km)');ylabel('z (km)');
% figure_FontSize=8;  
% set(get(gca,'XLabel'),'FontSize',figure_FontSize,'Vertical','middle'); 
% set(get(gca,'YLabel'),'FontSize',figure_FontSize,'Vertical','middle'); 
% set(findobj('FontSize',10),'FontSize',figure_FontSize);  
% set(findobj(get(gca,'Children'),'LineWidth',0.5),'LineWidth',2); 
% set(gcf, 'PaperPositionMode', 'manual');
% figure(3)
% imagesc(xx,zz,dp);colormap(gray);mx=max(max(p2));mn=min(min(p2));caxis([mn mx]/10000);set(gca,'xaxislocation','top'); xlabel('x /(km)');ylabel('z /(km)');colorbar;%title('vz seismogram ');
% set(gcf,'Position',[100 100 260 130]); colorbar;
% set(gca,'Position',[.13 .08 .680 .65]); colorbar('position',[0.83 0.08 0.04 0.6]);
% set(gca,'xaxislocation','top'); xlabel('x (km)');ylabel('z (km)');
% figure_FontSize=8;  
% set(get(gca,'XLabel'),'FontSize',figure_FontSize,'Vertical','middle'); 
% set(get(gca,'YLabel'),'FontSize',figure_FontSize,'Vertical','middle'); 
% set(findobj('FontSize',10),'FontSize',figure_FontSize);  
% set(findobj(get(gca,'Children'),'LineWidth',0.5),'LineWidth',2); 
% set(gcf, 'PaperPositionMode', 'manual');
%% actual model
fid=fopen('acc_vp.dat','rb');
fid2=fopen('acc_vs.dat','rb');                                                                                                                   
p1=fread(fid,[nz,nx],'float');
p2=fread(fid2,[nz,nx],'float');
fclose(fid);fclose(fid2);

accp=p1;
accs=p2;

figure(4)
imagesc(xx,zz,p1);mx=max(max(p1));mn=min(min(p1));caxis([mn mx]);
%set(gcf,'Position',[100 100 260 130]); %colorbar;
set(gcf,'Position',[100 300 500 200]);
set(gca,'Position',[.10 .08 .660 .65]); %colorbar('position',[0.83 0.08 0.04 0.6]);
h=colorbar('position',[0.78 0.08 0.04 0.65]);
set(get(h,'ylabel'),'string','Velocity (m/s)','FontSize',13);
 
set(gca,'xaxislocation','top'); xlabel('x (km)');ylabel('z (km)');
figure_FontSize=13;  
set(get(gca,'XLabel'),'FontSize',figure_FontSize); 
set(get(gca,'YLabel'),'FontSize',figure_FontSize); 
set(findobj('FontSize',13),'FontSize',figure_FontSize);  
set(findobj(get(gca,'Children'),'LineWidth',1),'LineWidth',2); 
set(gcf, 'PaperPositionMode', 'manual');
set(gca,'fontsize',13);
figure(5)
imagesc(xx,zz,p2);mx=max(max(p2));mn=min(min(p2));caxis([mn mx]);
%set(gcf,'Position',[100 100 260 130]); %colorbar;
set(gcf,'Position',[100 300 500 200]);
set(gca,'Position',[.10 .08 .660 .65]); %colorbar('position',[0.83 0.08 0.04 0.6]);
h=colorbar('position',[0.78 0.08 0.04 0.65]);
set(get(h,'ylabel'),'string','Velocity (m/s)','FontSize',13);
 
set(gca,'xaxislocation','top'); xlabel('x (km)');ylabel('z (km)');
%figure_FontSize=13;  
set(get(gca,'XLabel'),'FontSize',figure_FontSize); 
set(get(gca,'YLabel'),'FontSize',figure_FontSize); 
set(findobj('FontSize',13),'FontSize',figure_FontSize);  
set(findobj(get(gca,'Children'),'LineWidth',1),'LineWidth',2); 
set(gcf, 'PaperPositionMode', 'manual');
set(gca,'fontsize',13);
%% ini model
fid=fopen('ini_vp.dat','rb');
fid2=fopen('ini_vs.dat','rb');                                                                                                                  
p1=fread(fid,[nz,nx],'float');
p2=fread(fid2,[nz,nx],'float');
fclose(fid);fclose(fid2);
figure(6)
imagesc(xx,zz,p1);mx=max(max(p1));mn=min(min(p1));caxis([mn mx]);
%set(gcf,'Position',[100 100 260 130]); %colorbar;
set(gcf,'Position',[100 300 500 200]);
set(gca,'Position',[.10 .08 .660 .65]); %colorbar('position',[0.83 0.08 0.04 0.6]);
h=colorbar('position',[0.78 0.08 0.04 0.65]);
set(get(h,'ylabel'),'string','Velocity (m/s)','FontSize',13);
 
set(gca,'xaxislocation','top'); xlabel('x (km)');ylabel('z (km)');
figure_FontSize=13;  
set(get(gca,'XLabel'),'FontSize',figure_FontSize); 
set(get(gca,'YLabel'),'FontSize',figure_FontSize); 
set(findobj('FontSize',13),'FontSize',figure_FontSize);  
set(findobj(get(gca,'Children'),'LineWidth',1),'LineWidth',2); 
set(gcf, 'PaperPositionMode', 'manual');
set(gca,'fontsize',13);
figure(7)
imagesc(xx,zz,p2);mx=max(max(p2));mn=min(min(p2));caxis([mn mx]);
%set(gcf,'Position',[100 100 260 130]); %colorbar;
set(gcf,'Position',[100 300 500 200]);
set(gca,'Position',[.10 .08 .660 .65]); %colorbar('position',[0.83 0.08 0.04 0.6]);
h=colorbar('position',[0.78 0.08 0.04 0.65]);
set(get(h,'ylabel'),'string','Velocity (m/s)','FontSize',13);
 
set(gca,'xaxislocation','top'); xlabel('x (km)');ylabel('z (km)');
%figure_FontSize=13;  
set(get(gca,'XLabel'),'FontSize',figure_FontSize); 
set(get(gca,'YLabel'),'FontSize',figure_FontSize); 
set(findobj('FontSize',13),'FontSize',figure_FontSize);  
set(findobj(get(gca,'Children'),'LineWidth',1),'LineWidth',2); 
set(gcf, 'PaperPositionMode', 'manual');
set(gca,'fontsize',13);



inierrvp=(sum(sum((p1-accp).^2))/nx/nz).^0.5

inierrvs=(sum(sum((p2-accs).^2))/nx/nz).^0.5
%% inversion
fid=fopen('1ifreq_vp.dat','rb');
fid2=fopen('1ifreq_vs.dat','rb');                                                                                                                   
p1=fread(fid,[nz,nx],'float');
p2=fread(fid2,[nz,nx],'float');
fclose(fid);fclose(fid2);
figure(8)
imagesc(xx,zz,p1);mx=max(max(p1));mn=min(min(p1));caxis([mn mx]);
%set(gcf,'Position',[100 100 260 130]); %colorbar;
set(gcf,'Position',[100 300 500 200]);
set(gca,'Position',[.10 .08 .660 .65]); %colorbar('position',[0.83 0.08 0.04 0.6]);
h=colorbar('position',[0.78 0.08 0.04 0.65]);
set(get(h,'ylabel'),'string','Velocity (m/s)','FontSize',13);
 
set(gca,'xaxislocation','top'); xlabel('x (km)');ylabel('z (km)');
figure_FontSize=13;  
set(get(gca,'XLabel'),'FontSize',figure_FontSize); 
set(get(gca,'YLabel'),'FontSize',figure_FontSize); 
set(findobj('FontSize',13),'FontSize',figure_FontSize);  
set(findobj(get(gca,'Children'),'LineWidth',1),'LineWidth',2); 
set(gcf, 'PaperPositionMode', 'manual');
set(gca,'fontsize',13);
figure(9)
imagesc(xx,zz,p2);mx=max(max(p2));mn=min(min(p2));caxis([mn mx]);
%set(gcf,'Position',[100 100 260 130]); %colorbar;
set(gcf,'Position',[100 300 500 200]);
set(gca,'Position',[.10 .08 .660 .65]); %colorbar('position',[0.83 0.08 0.04 0.6]);
h=colorbar('position',[0.78 0.08 0.04 0.65]);
set(get(h,'ylabel'),'string','Velocity (m/s)','FontSize',13);
 
set(gca,'xaxislocation','top'); xlabel('x (km)');ylabel('z (km)');
%figure_FontSize=13;  
set(get(gca,'XLabel'),'FontSize',figure_FontSize); 
set(get(gca,'YLabel'),'FontSize',figure_FontSize); 
set(findobj('FontSize',13),'FontSize',figure_FontSize);  
set(findobj(get(gca,'Children'),'LineWidth',1),'LineWidth',2); 
set(gcf, 'PaperPositionMode', 'manual');
set(gca,'fontsize',13);
%% inversion
fid=fopen('3ifreq_vp.dat','rb');
fid2=fopen('3ifreq_vs.dat','rb');                                                                                                                   
p1=fread(fid,[nz,nx],'float');
p2=fread(fid2,[nz,nx],'float');
fclose(fid);fclose(fid2);
figure(10)
imagesc(xx,zz,p1);mx=max(max(p1));mn=min(min(p1));caxis([mn mx]);
%set(gcf,'Position',[100 100 260 130]); %colorbar;
set(gcf,'Position',[100 300 500 200]);
set(gca,'Position',[.10 .08 .660 .65]); %colorbar('position',[0.83 0.08 0.04 0.6]);
h=colorbar('position',[0.78 0.08 0.04 0.65]);
set(get(h,'ylabel'),'string','Velocity (m/s)','FontSize',13);
 
set(gca,'xaxislocation','top'); xlabel('x (km)');ylabel('z (km)');
figure_FontSize=13;  
set(get(gca,'XLabel'),'FontSize',figure_FontSize); 
set(get(gca,'YLabel'),'FontSize',figure_FontSize); 
set(findobj('FontSize',13),'FontSize',figure_FontSize);  
set(findobj(get(gca,'Children'),'LineWidth',1),'LineWidth',2); 
set(gcf, 'PaperPositionMode', 'manual');
set(gca,'fontsize',13);
figure(11)
imagesc(xx,zz,p2);mx=max(max(p2));mn=min(min(p2));caxis([mn mx]);
%set(gcf,'Position',[100 100 260 130]); %colorbar;
set(gcf,'Position',[100 300 500 200]);
set(gca,'Position',[.10 .08 .660 .65]); %colorbar('position',[0.83 0.08 0.04 0.6]);
h=colorbar('position',[0.78 0.08 0.04 0.65]);
set(get(h,'ylabel'),'string','Velocity (m/s)','FontSize',13);
 
set(gca,'xaxislocation','top'); xlabel('x (km)');ylabel('z (km)');
%figure_FontSize=13;  
set(get(gca,'XLabel'),'FontSize',figure_FontSize); 
set(get(gca,'YLabel'),'FontSize',figure_FontSize); 
set(findobj('FontSize',13),'FontSize',figure_FontSize);  
set(findobj(get(gca,'Children'),'LineWidth',1),'LineWidth',2); 
set(gcf, 'PaperPositionMode', 'manual');
set(gca,'fontsize',13);

%% inversion
fid=fopen('5ifreq_vp.dat','rb');
fid2=fopen('5ifreq_vs.dat','rb');                                                                                                                   
p1=fread(fid,[nz,nx],'float');
p2=fread(fid2,[nz,nx],'float');
fclose(fid);fclose(fid2);
figure(122)
imagesc(xx,zz,p1);mx=max(max(accp));mn=min(min(accp));caxis([mn mx]);
%set(gcf,'Position',[100 100 260 130]); %colorbar;
set(gcf,'Position',[100 300 500 200]);
set(gca,'Position',[.10 .08 .660 .65]); %colorbar('position',[0.83 0.08 0.04 0.6]);
h=colorbar('position',[0.78 0.08 0.04 0.65]);
set(get(h,'ylabel'),'string','Velocity (m/s)','FontSize',13);
 
set(gca,'xaxislocation','top'); xlabel('x (km)');ylabel('z (km)');
figure_FontSize=13;  
set(get(gca,'XLabel'),'FontSize',figure_FontSize); 
set(get(gca,'YLabel'),'FontSize',figure_FontSize); 
set(findobj('FontSize',13),'FontSize',figure_FontSize);  
set(findobj(get(gca,'Children'),'LineWidth',1),'LineWidth',2); 
set(gcf, 'PaperPositionMode', 'manual');
set(gca,'fontsize',13);
figure(133)
imagesc(xx,zz,p2);mx=max(max(accs));mn=min(min(accs));caxis([mn mx]);
%set(gcf,'Position',[100 100 260 130]); %colorbar;
set(gcf,'Position',[100 300 500 200]);
set(gca,'Position',[.10 .08 .660 .65]); %colorbar('position',[0.83 0.08 0.04 0.6]);
h=colorbar('position',[0.78 0.08 0.04 0.65]);
set(get(h,'ylabel'),'string','Velocity (m/s)','FontSize',13);
 
set(gca,'xaxislocation','top'); xlabel('x (km)');ylabel('z (km)');
%figure_FontSize=13;  
set(get(gca,'XLabel'),'FontSize',figure_FontSize); 
set(get(gca,'YLabel'),'FontSize',figure_FontSize); 
set(findobj('FontSize',13),'FontSize',figure_FontSize);  
set(findobj(get(gca,'Children'),'LineWidth',1),'LineWidth',2); 
set(gcf, 'PaperPositionMode', 'manual');
set(gca,'fontsize',13);
% %% obs seis
% fid=fopen('20source_seismogram_vz_obs.dat','rb');                                                                                                            
% p1=fread(fid,[nt,nx],'float');
% fclose(fid);
% figure(10)
% imagesc(xx,tt,p1);
% colormap(gray);mx=max(max(p1));mn=min(min(p1));caxis([mn mx]/15);
% set(gca,'xaxislocation','top'); %title('vx seismogram');
% set(gcf,'Position',[100 100 420 420]); 
% set(gca,'Position',[.13 .04 .680 .85]);
% colorbar('position',[0.83 0.040 0.04 0.8]);set(gca, 'CLim', [-6*10^-9 6*10^-9]);
% %set(gca,'xaxislocation','top'); 
% xlabel('x (km)');ylabel('t (s)');
% figure_FontSize=13;  
% set(get(gca,'XLabel'),'FontSize',figure_FontSize); 
% set(get(gca,'YLabel'),'FontSize',figure_FontSize); 
% set(findobj('FontSize',13),'FontSize',figure_FontSize);  
% set(findobj(get(gca,'Children'),'LineWidth',0.5),'LineWidth',2); 
% set(gcf, 'PaperPositionMode', 'manual');
%set(gca,'fontsize',13);

% %% syn seis
% fid=fopen('20source_seismogram_vz_10syn.dat','rb');                                                                                                            
% p1=fread(fid,[nt,nx],'float');
% fclose(fid);
% pcc=p1;
% figure(11)
% imagesc(xx,tt,p1);
% colormap(gray);mx=max(max(pcc));mn=min(min(pcc));caxis([mn mx]/40);
% set(gca,'xaxislocation','top'); %title('vx seismogram');
% set(gcf,'Position',[100 100 420 420]); 
% set(gca,'Position',[.13 .04 .680 .85]);
% colorbar('position',[0.83 0.040 0.04 0.8]);set(gca, 'CLim', [-6*10^-9 6*10^-9]);
% %set(gca,'xaxislocation','top'); 
% xlabel('x (km)');ylabel('t (s)');
% figure_FontSize=13;  
% set(get(gca,'XLabel'),'FontSize',figure_FontSize); 
% set(get(gca,'YLabel'),'FontSize',figure_FontSize); 
% set(findobj('FontSize',13),'FontSize',figure_FontSize);  
% set(findobj(get(gca,'Children'),'LineWidth',0.5),'LineWidth',2); 
% set(gcf, 'PaperPositionMode', 'manual');
% set(gca,'fontsize',13);
% 
% 
% %% rms seis
% fid=fopen('20source_seismogram_vz_10rms.dat','rb');                                                                                                            
% p1=fread(fid,[nt,nx],'float');
% fclose(fid);
% figure(12)
% imagesc(xx,tt,p1);
% colormap(gray);mx=max(max(pcc));mn=min(min(pcc));caxis([mn mx]);
% set(gca,'xaxislocation','top'); %title('vx seismogram');
% set(gcf,'Position',[100 100 420 420]); 
% set(gca,'Position',[.13 .04 .680 .85]);
% colorbar('position',[0.83 0.040 0.04 0.8]);%set(gca, 'CLim', [-6*10^-9 6*10^-9]);
% %set(gca,'xaxislocation','top'); 
% xlabel('x (km)');ylabel('t (s)');
% figure_FontSize=13;  
% set(get(gca,'XLabel'),'FontSize',figure_FontSize); 
% set(get(gca,'YLabel'),'FontSize',figure_FontSize); 
% set(findobj('FontSize',13),'FontSize',figure_FontSize);  
% set(findobj(get(gca,'Children'),'LineWidth',0.5),'LineWidth',2); 
% set(gcf, 'PaperPositionMode', 'manual');
% set(gca,'fontsize',13);
% 
% 
% %% MIsfit
% n1=10;
% n2=10;
% n3=10;
% n4=10;
% n5=10;
% 
% [L1] = textread('misfit_1ifreq.txt', '%17f', -1);
% [L2] = textread('misfit_2ifreq.txt', '%17f', -1);
% [L3] = textread('misfit_3ifreq.txt', '%17f', -1);
% [L4] = textread('misfit_4ifreq.txt', '%17f', -1);
% [L5] = textread('misfit_5ifreq.txt', '%17f', -1);
% % 
% % L1=L1/max(L1);
% % L2=L2/max(L1);
% % L3=L3/max(L1);
% % L4=L4/max(L1);
% % L5=L5/max(L1);
% 
% L=[L1;L2;L3;L4;L5];
% L=L*10^-4;
% Lx=[1:10,1:10,1:10,1:10,1:10]';
% 
% figure;
% 
% plot(L,'r*','markersize',4);hold on;
% xlabel('Iterations','fontsize',13);
% ylabel('Objective function misfit','fontsize',13);
% 
% 
% plot([10.5 10.5],[0 3],'b--');
% plot([20.5 20.5],[0 3],'b--');
% plot([30.5 30.5],[0 3],'b--');
% plot([40.5 40.5],[0 3],'b--');
% 
% axis([1 50 0 3]);
% 
% gtext('4Hz','fontsize',13);
% gtext('7Hz','fontsize',13);
% gtext('9Hz','fontsize',13);
% gtext('14Hz','fontsize',13);
% gtext('18Hz','fontsize',13);
% 
% set(gcf,'Position',[100 500 450 300]);
% 
% 
% fid=fopen('5ifreq_vp.dat','rb');
% fid2=fopen('5ifreq_vs.dat','rb');                                                                                                                   
% p1=fread(fid,[nz,nx],'float');
% p2=fread(fid2,[nz,nx],'float');
% fclose(fid);fclose(fid2);
% figure(8)
% imagesc(xx,zz,p1);mx=max(max(p1));mn=min(min(p1));caxis([mn mx]);
% %set(gcf,'Position',[100 100 260 130]); %colorbar;
% set(gcf,'Position',[100 300 500 500]);
% set(gca,'Position',[.12 .08 .660 .75]); %colorbar('position',[0.83 0.08 0.04 0.6]);
% h=colorbar('position',[0.80 0.08 0.04 0.75]);
% set(get(h,'ylabel'),'string','Velocity (m/s)','FontSize',13);
%  
% set(gca,'xaxislocation','top'); xlabel('x (km)');ylabel('z (km)');
% figure_FontSize=13;  
% set(get(gca,'XLabel'),'FontSize',figure_FontSize); 
% set(get(gca,'YLabel'),'FontSize',figure_FontSize); 
% set(findobj('FontSize',13),'FontSize',figure_FontSize);  
% set(findobj(get(gca,'Children'),'LineWidth',1),'LineWidth',2); 
% set(gcf, 'PaperPositionMode', 'manual');
% set(gca,'fontsize',13);
% 
% figure(9)
% imagesc(xx,zz,p2);mx=max(max(p1));mn=min(min(p1));caxis([mn mx]);
% %set(gcf,'Position',[100 100 260 130]); %colorbar;
% set(gcf,'Position',[100 300 500 500]);
% set(gca,'Position',[.12 .08 .660 .75]); %colorbar('position',[0.83 0.08 0.04 0.6]);
% h=colorbar('position',[0.80 0.08 0.04 0.75]);
% set(get(h,'ylabel'),'string','Velocity (m/s)','FontSize',13);
%  
% set(gca,'xaxislocation','top'); xlabel('x (km)');ylabel('z (km)');
% figure_FontSize=13;  
% set(get(gca,'XLabel'),'FontSize',figure_FontSize); 
% set(get(gca,'YLabel'),'FontSize',figure_FontSize); 
% set(findobj('FontSize',13),'FontSize',figure_FontSize);  
% set(findobj(get(gca,'Children'),'LineWidth',1),'LineWidth',2); 
% set(gcf, 'PaperPositionMode', 'manual');
% set(gca,'fontsize',13);
% 
% sum1=0;
% for i=1:nx
%     for j=1:nz
%             sum1=sum1+abs((accp(j,i)-p1(j,i)))/accp(j,i);
%     end
% end
% sum1/nx/nz
% 
% sum2=0;
% for i=1:nx
%     for j=1:nz
%             sum2=sum2+abs((accs(j,i)-p2(j,i)))/accs(j,i);
%     end
% end
% sum2/nx/nz
% 
% pl=p1(90:130,400:600);
% zl=zz(90:130);
% xl=xx(400:600);
% 
% figure(9)
% imagesc(xl,zl,pl);mx=max(max(pl));mn=min(min(pl));caxis([mn mx]);
% %set(gcf,'Position',[100 100 260 130]); %colorbar;
% set(gcf,'Position',[100 300 500 300]);
% set(gca,'Position',[.12 .08 .660 .75]); %colorbar('position',[0.83 0.08 0.04 0.6]);
% h=colorbar('position',[0.80 0.08 0.04 0.75]);
% set(get(h,'ylabel'),'string','Velocity (m/s)','FontSize',13);
% set(gca, 'CLim', [2600 3500]);
% set(gca,'xaxislocation','top'); xlabel('x (km)');ylabel('z (km)');
% figure_FontSize=13;  
% set(get(gca,'XLabel'),'FontSize',figure_FontSize); 
% set(get(gca,'YLabel'),'FontSize',figure_FontSize); 
% set(findobj('FontSize',13),'FontSize',figure_FontSize);  
% set(findobj(get(gca,'Children'),'LineWidth',1),'LineWidth',2); 
% set(gcf, 'PaperPositionMode', 'manual');
% set(gca,'fontsize',13);


errvp=(sum(sum((p1-accp).^2))/nx/nz).^0.5

errvs=(sum(sum((p2-accs).^2))/nx/nz).^0.5