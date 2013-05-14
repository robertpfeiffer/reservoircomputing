from PyQt4 import QtGui, QtCore
from matplotlib import pylab
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
import sys
import time

from flight_data import *

class Data_Visualization_GUI(object):
    def __init__(self,Data_Visualization):
        object.__init__(self)
        
        Data_Visualization.resize(1000,650)
        Data_Visualization.setWindowTitle("Didi's Data Visualizer")
        
        self.mainLayout = QtGui.QGridLayout(Data_Visualization)
        self.mainLayout.setSpacing(5)
        
        self.trackingPlotWidget = QtGui.QWidget()
        self.fig = pylab.figure()
        self.axes = pylab.Axes(self.fig, [.1,.1,.8,.8])
        self.fig.add_axes(self.axes)   
        self.plot, = pylab.plot([0],[0],'ro')
        self.axes.set_xlim(-5, 5)
        self.axes.set_ylim(-5, 5)
        self.plotTitle = pylab.title('Tracking Data')
        self.plotxLabel = pylab.xlabel('X')
        self.plotyLabel = pylab.ylabel('Z')
        self.canvas = FigureCanvas(self.fig)       
        self.canvas.setParent(self.trackingPlotWidget)
        self.canvas.setFixedSize(400,400)
        self.mpl_toolbar = NavigationToolbar(self.canvas, self.trackingPlotWidget)

##
##
##        self.XYZPlotWidget = QtGui.QWidget()
##        pylab.subplot(311)
##        self.fig2 = pylab.figure()
##        self.axes2 = pylab.Axes(self.fig2, [.1,.1,.8,.8])
##        self.fig2.add_axes(self.axes2)   
##        self.plot2 = pylab.plot([0],[0],'r', [0],[0],'g', [0],[0],'b')
##        self.axes2.set_xlim(-200, 0)
##        self.axes2.set_ylim(-5, 5)
##        pylab.legend([self.plot2[0], self.plot2[1], self.plot2[2]], ["X", "Y", "Z"], loc=3)
##        self.plotxLabel2 = pylab.xlabel('Time')
##        self.plotyLabel2 = pylab.ylabel('XYZ')
##        self.canvas2 = FigureCanvas(self.fig2)       
##        self.canvas2.setParent(self.XYZPlotWidget)
##        self.canvas2.setFixedSize(500,150)
##        
##        pylab.subplot(312)
##        self.fig3 = pylab.figure()
##        self.axes3 = pylab.Axes(self.fig3, [.1,.1,.8,.8])
##        self.fig3.add_axes(self.axes3)   
##        self.plot3 = pylab.plot([0],[0],'g', [0],[0],'r')
##        self.axes3.set_xlim(-200, 0)
##        self.axes3.set_ylim(-12, 12)
##        pylab.legend([self.plot3[0], self.plot3[1]], ["Pitch", "Roll"], loc=3)
##        self.plotxLabel3 = pylab.xlabel('Time')
##        self.plotyLabel3 = pylab.ylabel('PR')
##        self.canvas3 = FigureCanvas(self.fig3)       
##        self.canvas3.setParent(self.XYZPlotWidget)
##        self.canvas3.setFixedSize(500,150)
##        
##        pylab.subplot(313)
##        self.fig4 = pylab.figure()
##        self.axes4 = pylab.Axes(self.fig4, [.1,.1,.8,.8])
##        self.fig4.add_axes(self.axes4)   
##        self.plot4 = pylab.plot([0],[0],'r', [0],[0],'g', [0],[0],'b', [0],[0],'y')
##        pylab.legend([self.plot4[0], self.plot4[1], self.plot4[2], self.plot4[3]], ["w1", "w2", "w3", "w4"], loc=3)
##        self.axes4.set_xlim(-200, 0)
##        self.axes4.set_ylim(-0.5, 0.5)
##        self.plotxLabel4 = pylab.xlabel('Time')
##        self.plotyLabel4 = pylab.ylabel('WWWW')
##        self.canvas4 = FigureCanvas(self.fig4)       
##        self.canvas4.setParent(self.XYZPlotWidget)
##        self.canvas4.setFixedSize(500,150)                  


        self.XYZPlotWidget = QtGui.QWidget()
        self.fig2 = pylab.figure()
        self.axes2 = pylab.Axes(self.fig2, [.1,.1,.8,.8])
        self.fig2.add_axes(self.axes2)   
        self.plot2 = pylab.plot([0],[0],'r', [0],[0],'g', [0],[0],'b')
        self.axes2.set_xlim(-200, 0)
        self.axes2.set_ylim(-5, 5)
        pylab.legend([self.plot2[0], self.plot2[1], self.plot2[2]], ["X", "Y", "Z"], loc=3)
#        self.plotTitle2 = pylab.title('XYZ Data')
        self.plotxLabel2 = pylab.xlabel('Time')
        self.plotyLabel2 = pylab.ylabel('XYZ')
        self.canvas2 = FigureCanvas(self.fig2)       
        self.canvas2.setParent(self.XYZPlotWidget)
        self.canvas2.setFixedSize(500,150)
        
        self.PRPlotWidget = QtGui.QWidget()
        self.fig3 = pylab.figure()
        self.axes3 = pylab.Axes(self.fig3, [.1,.1,.8,.8])
        self.fig3.add_axes(self.axes3)   
        self.plot3 = pylab.plot([0],[0],'g', [0],[0],'r')
        self.axes3.set_xlim(-200, 0)
        self.axes3.set_ylim(-12, 12)
        pylab.legend([self.plot3[0], self.plot3[1]], ["Pitch", "Roll"], loc=3)
#        self.plotTitle3 = pylab.title('Pitch/Roll Data')
        self.plotxLabel3 = pylab.xlabel('Time')
        self.plotyLabel3 = pylab.ylabel('PR')
        self.canvas3 = FigureCanvas(self.fig3)       
        self.canvas3.setParent(self.PRPlotWidget)
        self.canvas3.setFixedSize(500,150)
        
        self.WWWWPlotWidget = QtGui.QWidget()
        self.fig4 = pylab.figure()
        self.axes4 = pylab.Axes(self.fig4, [.1,.1,.8,.8])
        self.fig4.add_axes(self.axes4)   
        self.plot4 = pylab.plot([0],[0],'r', [0],[0],'g', [0],[0],'b', [0],[0],'y')
        pylab.legend([self.plot4[0], self.plot4[1], self.plot4[2], self.plot4[3]], ["w1", "w2", "w3", "w4"], loc=3)
        self.axes4.set_xlim(-200, 0)
        self.axes4.set_ylim(-0.5, 0.5)
#        self.plotTitle4 = pylab.title('WWWW Data')
        self.plotxLabel4 = pylab.xlabel('Time')
        self.plotyLabel4 = pylab.ylabel('WWWW')
        self.canvas4 = FigureCanvas(self.fig4)       
        self.canvas4.setParent(self.WWWWPlotWidget)
        self.canvas4.setFixedSize(500,150)                
        
        self.lblX = QtGui.QLabel('X: ')
        self.lblY = QtGui.QLabel('Y: ')
        self.lblZ = QtGui.QLabel('Z: ')
        
        self.lblYaw = QtGui.QLabel('Yaw: ')
        self.lblPitch = QtGui.QLabel('Pitch: ')
        self.lblRoll = QtGui.QLabel('Roll: ')
        self.lblAltitude = QtGui.QLabel('Altitude: ')
        
        self.lblVX = QtGui.QLabel('Speed X: ')
        self.lblVY = QtGui.QLabel('Speed Y: ')
        self.lblVZ = QtGui.QLabel('Speed Z: ')
        
        self.lblTime = QtGui.QLabel('Time: ')
        self.lblCommand = QtGui.QLabel('Command: ')
        self.lblBattery = QtGui.QLabel('Battery: ')
        
        self.btnLoad = QtGui.QPushButton('Load')
        self.btnReplay = QtGui.QPushButton('Replay')

        self.mainLayout.addWidget(self.btnLoad,1,1)
        self.mainLayout.addWidget(self.btnReplay,1,2)
        
        self.mainLayout.addWidget(self.lblX,2,0)
        self.mainLayout.addWidget(self.lblY,2,1)
        self.mainLayout.addWidget(self.lblZ,2,2)
        
        self.mainLayout.addWidget(self.lblYaw,3,0)
        self.mainLayout.addWidget(self.lblPitch,3,1)
        self.mainLayout.addWidget(self.lblRoll,3,2)
        self.mainLayout.addWidget(self.lblAltitude,3,3)
        
        self.mainLayout.addWidget(self.lblVX,4,0)
        self.mainLayout.addWidget(self.lblVY,4,1)
        self.mainLayout.addWidget(self.lblVZ,4,2)
        
        self.mainLayout.addWidget(self.lblTime,5,0)
        self.mainLayout.addWidget(self.lblCommand,5,1)
        self.mainLayout.addWidget(self.lblBattery,5,2)
        
        self.mainLayout.addWidget(self.mpl_toolbar,6,0,1,4)
        self.mainLayout.addWidget(self.trackingPlotWidget,7,0,5,4)
        self.mainLayout.addWidget(self.XYZPlotWidget,2,5,6,4)
        self.mainLayout.addWidget(self.PRPlotWidget,8,5,2,4)
        self.mainLayout.addWidget(self.WWWWPlotWidget,10,5,2,4)
        
        for i in range(self.mainLayout.count()): 
            w = self.mainLayout.itemAt(i).widget()
            if type(w)==QtGui.QLabel: 
                w.setFixedSize(100, 15)
                
        
        QtCore.QObject.connect(self.btnReplay, QtCore.SIGNAL('clicked()'), Data_Visualization.visualize)
        QtCore.QObject.connect(self.btnLoad, QtCore.SIGNAL('clicked()'), Data_Visualization.open_file)

class Data_Visualization(QtGui.QWidget, FlightData):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        #filename = QtGui.QFileDialog.getOpenFileName(self, 'Open File')
        #filename = '/Users/witali/Daten/CogSci/StudyProject/Ergebnisse/flight_Wed_13_Feb_2013_14_25_11_AllDataXXX1'
        filename = 'flight_data/flight_random_points_with_target/flight_Wed_06_Feb_2013_16_07_34_AllData'
        FlightData.__init__(self, filename)
        
        #self.initVariables()
        self.initGui()
        #self.loadData()
        
    def open_file(self):
        filename = QtGui.QFileDialog.getOpenFileName(self, 'Open File')
        self.load_file(filename)
                
    def initGui(self):
        self.ui = Data_Visualization_GUI(self)
        self.show()
    
    def visualize(self):
        for i in range(1,len(self.dataAltitude),10):
            try:
                self.ui.lblAltitude.setText('Altitude: ' + str(self.dataAltitude[i]))
            except:
                print 'Altitude Error'
            self.ui.lblBattery.setText('Battery: ' + self.dataBat[i])
            self.ui.lblCommand.setText('Command: ' + self.dataCommand[i])
            self.ui.lblPitch.setText('Pitch: ' + str(self.dataPitch[i]))
            self.ui.lblRoll.setText('Roll: ' + str(self.dataRoll[i]))
            self.ui.lblTime.setText('Time: ' + self.dataTime[i])
            self.ui.lblVX.setText('Speed X: ' + str(self.dataVX[i]))
            self.ui.lblVY.setText('Speed Y: ' + str(self.dataVY[i]))
            self.ui.lblVZ.setText('Speed Z: ' + str(self.dataVZ[i]))
            self.ui.lblX.setText('X: ' + str(self.dataX[i]))
            self.ui.lblY.setText('Y: ' + str(self.dataY[i]))
            self.ui.lblYaw.setText('Yaw: ' + str(self.dataYaw[i]))
            self.ui.lblZ.setText('Z: ' + str(self.dataZ[i]))
            self.ui.plot.set_data([self.dataX[i-min(200,i):i]], [self.dataZ[i-min(200,i):i]])
            self.ui.canvas.draw()
            self.ui.plot2[0].set_data(range(-min(200,i),0),[self.dataX[i-min(200,i):i]])
            self.ui.plot2[1].set_data(range(-min(200,i),0),[self.dataY[i-min(200,i):i]])
            self.ui.plot2[2].set_data(range(-min(200,i),0),[self.dataZ[i-min(200,i):i]])
            self.ui.canvas2.draw()

            self.ui.plot3[0].set_data(range(-min(200,i),0),[self.dataPitch[i-min(200,i):i]])
            self.ui.plot3[1].set_data(range(-min(200,i),0),[self.dataRoll[i-min(200,i):i]])
            self.ui.canvas3.draw()

            self.ui.plot4[0].set_data(range(-min(200,i),0),[self.dataw1[i-min(200,i):i]])
            self.ui.plot4[1].set_data(range(-min(200,i),0),[self.dataw2[i-min(200,i):i]])
            self.ui.plot4[2].set_data(range(-min(200,i),0),[self.dataw3[i-min(200,i):i]])
            self.ui.plot4[3].set_data(range(-min(200,i),0),[self.dataw4[i-min(200,i):i]])
            self.ui.canvas4.draw()
            
            QtGui.QApplication.processEvents()
            self.repaint()
            time.sleep(0.01)
        
        
if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    v = Data_Visualization()  
    sys.exit(app.exec_())

