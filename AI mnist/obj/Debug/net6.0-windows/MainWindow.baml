   M S B A M L   `   `   `  ����C  ?AI mnist, Version=1.0.0.0, Culture=neutral, PublicKeyToken=null#clr-namespace:AI_mnistAI_mnist  5   	       AI_mnist.MainWindow   5   	   W SPresentationCore, Version=6.0.2.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35\ XPresentationFramework, Version=6.0.2.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35e aSystem.Windows.Controls.Ribbon, Version=6.0.2.0, Culture=neutral, PublicKeyToken=b77a5c561934e089R NWindowsBase, Version=6.0.2.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35^ ZWindowsFormsIntegration, Version=6.0.2.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35H 9http://schemas.microsoft.com/winfx/2006/xaml/presentation      5   	   R NSystem.Xaml, Version=6.0.2.0, Culture=neutral, PublicKeyToken=b77a5c561934e089:x,http://schemas.microsoft.com/winfx/2006/xaml     5   	   @d2http://schemas.microsoft.com/expression/blend/2008     5   	   Jmc;http://schemas.openxmlformats.org/markup-compatibility/2006     5   	   (localclr-namespace:AI_mnist     5   	     � Title$  
MainWindow��5   	   $	��450��6   $	��800��6,   $	��450��6;   $	��800��6H    � WindowStartupLocation$ CenterScreen=�6T   .��5	      �  � RowDefinitions 5
   
   �� 5      $
��0.3* �6   �� 5      $��* �6   �� 5      $
��0.3* �6   .+�5   
   ��  �� Orientation$ 
Horizontal=�6   .+�5      �� -   5        $��IterationTextBlock��5      $	��200��62   $��20, 20q�6A   �� 5      -   5        $��CorrectionTextBlock��5      $	��200��63   $��20, 20q�6B   5      � 5   
   $��1��6    � ColumnDefinitions 5      �� 5      $��* �6#   �� 5      $��* �6#   .+�5      �� -   5        $��
DigitImage��5      �� 5      -   5        $��
ResultText��5      $��1��6*   $	��100(�6:   $��Center=�6I   $��Center=�6d   5      �� 5   
   $��2��6   $ 
Horizontal=�6"   $��Center=�6;   .+�5       �� -   5        $��Распознать��5       $��1��6*   $��25(�67   $��10q�6Z   $��5q�6f   �� 5!      -   5        $��TrainButton��5!   '   $��Обучить��6   $��1��6:   $��25(�6G   $��10q�6o   $��5q�6{   5"      5#      5$      