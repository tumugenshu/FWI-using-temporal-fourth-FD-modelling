clear;
clear all;
clc;
nz=240;
nx=1000;
nt=2310;
dx=15;dz=15;dt=0.0018;
xx=0:dx:dx*(nx-1);
zz=0:dz:dz*(nz-1);
xx=xx/1000;
zz=zz/1000;
tt=0:dt:dt*(nt-1);

%% actual model
fid=fopen('acc_vp.dat','rb');
fid2=fopen('acc_vs.dat','rb');                                                                                                                   
p1=fread(fid,[nz,nx],'float');
p2=fread(fid2,[nz,nx],'float');
fclose(fid);fclose(fid2);
p1=p1/1000;
p2=p2/1000;

% maxp1=max(max(abs(p1)));
% maxp2=max(max(abs(p2)));
% p1=p1/maxp1;
% p2=p2/maxp1;

p_acc=p1(:,350);
s_acc=p2(:,350);

p_acc=p1(:,250);;
s_acc=p2(:,250);;


fid=fopen('ini_vp.dat','rb');
fid2=fopen('ini_vs.dat','rb');                                                                                                                   
p1=fread(fid,[nz,nx],'float');
p2=fread(fid2,[nz,nx],'float');
fclose(fid);fclose(fid2);
p1=p1/1000;
p2=p2/1000;

p_ini=p1(:,350);
s_ini=p2(:,350);

p_ini=p1(:,250);;
s_ini=p2(:,250);;
%% inversion
fid=fopen('5ifreq_vp.dat','rb');
fid2=fopen('5ifreq_vs.dat','rb');                                                                                                                   
p1=fread(fid,[nz,nx],'float');
p2=fread(fid2,[nz,nx],'float');
fclose(fid);fclose(fid2);

% maxp1=max(max(abs(p1)));
% maxp2=max(max(abs(p2)));
p1=p1/1000;
p2=p2/1000;


p=p1(:,250);
s=p2(:,250);

fid=fopen('5ifreq_vpn.dat','rb');
fid2=fopen('5ifreq_vsn.dat','rb');                                                                                                                   
p1=fread(fid,[nz,nx],'float');
p2=fread(fid2,[nz,nx],'float');
fclose(fid);fclose(fid2);
p1=p1/1000;
p2=p2/1000;
% maxp1=max(max(abs(p1)));
% maxp2=max(max(abs(p2)));
% p1=p1/maxp1;
% p2=p2/maxp1;

p_cov=p1(:,250);;
s_cov=p2(:,250);;

%%

figure(1)
h1=axes('position',[0.15 0.15 0.85 .85]);
set(gcf,'Position',[100 300 560 320]);
set(gca,'Position',[.15 .18 .80 .75]);
axis(h1);
h3=plot(zz,p_acc,'-k',zz,p_ini,'-g',zz,p,'--r',zz,p_cov,'--b');
xlabel('Depth (km)','fontsize',13);
ylabel('P-velocity (km/s)','fontsize',13);
set(gca,'fontsize',13);
axis([0 3.6 2.800 4.200])
legend('Actual','Initial','T2','T4');

% h2=axes('position',[0.21 0.55 0.34 0.36]);
% axis(h2);
% 
%         figure_FontSize=13;  
%         set(get(gca,'XLabel'),'FontSize',figure_FontSize,'Vertical','middle'); 
%         set(get(gca,'YLabel'),'FontSize',figure_FontSize,'Vertical','middle'); 
%         set(findobj('FontSize',10),'FontSize',figure_FontSize);  
%         set(findobj(get(gca,'Children'),'LineWidth',1),'LineWidth',1); 
%         set(gcf, 'PaperPositionMode', 'manual');
% h4=plot(zz,p_acc,'-k',zz,p_ini,'-g',zz,p,'--r',zz,p_cov,'--b');
% axis([0.2 0.55 1.9 2.300]);
% hold on 
% h=[h3;h4];

set(h1,'LineWidth',1.5); 


figure(2)
h1=axes('position',[0.15 0.15 0.85 .85]);
set(gcf,'Position',[100 300 560 320]);
set(gca,'Position',[.15 .18 .80 .75]);
axis(h1);
h3=plot(zz,s_acc,'-k',zz,s_ini,'-g',zz,s,'--r',zz,s_cov,'--b');
xlabel('Depth (km)','fontsize',13);
ylabel('S-velocity (km/s)','fontsize',13);
set(gca,'fontsize',13);
axis([0 3.6 1.600 2.60])

% h2=axes('position',[0.21 0.55 0.34 0.36]);
% axis(h2);
% 
%         figure_FontSize=13;  
%         set(get(gca,'XLabel'),'FontSize',figure_FontSize,'Vertical','middle'); 
%         set(get(gca,'YLabel'),'FontSize',figure_FontSize,'Vertical','middle'); 
%         set(findobj('FontSize',10),'FontSize',figure_FontSize);  
%         set(findobj(get(gca,'Children'),'LineWidth',1),'LineWidth',1); 
%         set(gcf, 'PaperPositionMode', 'manual');
% h4=plot(zz,s_acc,'-k',zz,s_ini,'-g',zz,s,'--r',zz,s_cov,'--b');
% axis([0.5 0.88 1.25 1.7500]);
% hold on 
% h=[h3;h4];

set(h1,'LineWidth',1.5); 
legend('Actual','Initial','T2','T4');

%% actual model
fid=fopen('acc_vp.dat','rb');
fid2=fopen('acc_vs.dat','rb');                                                                                                                   
p1=fread(fid,[nz,nx],'float');
p2=fread(fid2,[nz,nx],'float');
fclose(fid);fclose(fid2);
p1=p1/1000;
p2=p2/1000;

% maxp1=max(max(abs(p1)));
% maxp2=max(max(abs(p2)));
% p1=p1/maxp1;
% p2=p2/maxp1;

p_acc=p1(:,350);
s_acc=p2(:,350);



fid=fopen('ini_vp.dat','rb');
fid2=fopen('ini_vs.dat','rb');                                                                                                                   
p1=fread(fid,[nz,nx],'float');
p2=fread(fid2,[nz,nx],'float');
fclose(fid);fclose(fid2);
p1=p1/1000;
p2=p2/1000;

p_ini=p1(:,350);
s_ini=p2(:,350);


%% inversion
fid=fopen('4ifreq_vp.dat','rb');
fid2=fopen('4ifreq_vs.dat','rb');                                                                                                                   
p1=fread(fid,[nz,nx],'float');
p2=fread(fid2,[nz,nx],'float');
fclose(fid);fclose(fid2);

% maxp1=max(max(abs(p1)));
% maxp2=max(max(abs(p2)));
p1=p1/1000;
p2=p2/1000;


p=p1(:,350);
s=p2(:,350);


fid=fopen('4ifreq_vpn.dat','rb');
fid2=fopen('4ifreq_vsn.dat','rb');                                                                                                                   
p1=fread(fid,[nz,nx],'float');
p2=fread(fid2,[nz,nx],'float');
fclose(fid);fclose(fid2);
p1=p1/1000;
p2=p2/1000;
% maxp1=max(max(abs(p1)));
% maxp2=max(max(abs(p2)));
% p1=p1/maxp1;
% p2=p2/maxp1;


p_cov=p1(:,350);
s_cov=p2(:,350);



%%

figure(3)
h1=axes('position',[0.15 0.15 0.85 .85]);
set(gcf,'Position',[100 300 560 320]);
set(gca,'Position',[.15 .18 .80 .75]);
axis(h1);
h3=plot(zz,p_acc,'-k',zz,p_ini,'-g',zz,p,'--r',zz,p_cov,'--b');
xlabel('Depth (km)','fontsize',13);
ylabel('P-velocity (km/s)','fontsize',13);
set(gca,'fontsize',13);
axis([0 3.6 2.800 4.2])

% h2=axes('position',[0.21 0.55 0.29 0.36]);
% axis(h2);
% 
%         figure_FontSize=13;  
%         set(get(gca,'XLabel'),'FontSize',figure_FontSize,'Vertical','middle'); 
%         set(get(gca,'YLabel'),'FontSize',figure_FontSize,'Vertical','middle'); 
%         set(findobj('FontSize',10),'FontSize',figure_FontSize);  
%         set(findobj(get(gca,'Children'),'LineWidth',1),'LineWidth',1); 
%         set(gcf, 'PaperPositionMode', 'manual');
% h4=plot(zz,p_acc,'-k',zz,p_ini,'-g',zz,p,'--r',zz,p_cov,'--b');
% axis([0.2 0.45 1.9 2.800]);
% hold on 
% h=[h3;h4];

set(h1,'LineWidth',1.5); 
legend('Actual','Initial','T2','T4');

figure(4)
h1=axes('position',[0.15 0.15 0.85 .85]);
set(gcf,'Position',[100 300 560 320]);
set(gca,'Position',[.15 .18 .80 .75]);
axis(h1);
h3=plot(zz,s_acc,'-k',zz,s_ini,'-g',zz,s,'--r',zz,s_cov,'--b');
xlabel('Depth (km)','fontsize',13);
ylabel('S-velocity (km/s)','fontsize',13);
set(gca,'fontsize',13);
axis([0 3.6 1.600 2.60])

% h2=axes('position',[0.21 0.55 0.29 0.36]);
% axis(h2);
% 
%         figure_FontSize=13;  
%         set(get(gca,'XLabel'),'FontSize',figure_FontSize,'Vertical','middle'); 
%         set(get(gca,'YLabel'),'FontSize',figure_FontSize,'Vertical','middle'); 
%         set(findobj('FontSize',10),'FontSize',figure_FontSize);  
%         set(findobj(get(gca,'Children'),'LineWidth',1),'LineWidth',1); 
%         set(gcf, 'PaperPositionMode', 'manual');
% h4=plot(zz,s_acc,'-k',zz,s_ini,'-g',zz,s,'--r',zz,s_cov,'--b');
% axis([0.5 0.82 1.4 1.7500]);
% hold on 
% h=[h3;h4];
legend('Actual','Initial','T2','T4');
set(h1,'LineWidth',1.5); 