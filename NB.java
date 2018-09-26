package com.NaiveBayes;

import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Instances;
import java.util.Scanner;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import java.io.File;
import javax.swing.JFileChooser;
import javax.swing.filechooser.FileSystemView;
public class NB
{
	Instances trainDataset,testDataset;
	NaiveBayes nbModel;
	Evaluation ev;
	void getData(String FP) throws Exception
	{
		DataSource source = new DataSource(FP);
		Instances dataset = source.getDataSet();
		//set class index to the last attribute
		dataset.setClassIndex(dataset.numAttributes()-1);
		// randomize instance order before splitting dataset
		dataset.randomize(new java.util.Random());
		int trainSize = (int) Math.round(dataset.numInstances() * 0.7);
        int testSize = dataset.numInstances() - trainSize;
		trainDataset = new Instances(dataset, 0, trainSize);
        testDataset = new Instances(dataset, trainSize, testSize);
		System.out.println("File read successfully!");
	}
	void buildClassifier() throws Exception
	{
		nbModel = new NaiveBayes();
		nbModel.buildClassifier(trainDataset);
	}
	void Evaluate() throws Exception
	{
		ev = new Evaluation(trainDataset);
		ev.evaluateModel(nbModel, testDataset);

        System.out.println(ev.toSummaryString("Summary::", false));
        System.out.println(ev.toClassDetailsString("===============================\nClass wise detailed metrics::"));
        System.out.println(ev.toMatrixString("===============================\nConfusion Matrix"));
	}
	public static void main(String args[]) throws Exception
	{
		String fp = new String();
		JFileChooser jfc=new JFileChooser(FileSystemView.getFileSystemView().getHomeDirectory());
		int returnVal = jfc.showOpenDialog(null);
		if(returnVal == JFileChooser.APPROVE_OPTION)
		{
			fp = jfc.getSelectedFile().getAbsolutePath();
		}
		NB nb=new NB();
		nb.getData(fp);
		nb.buildClassifier();
		nb.Evaluate();
	}
}


OUTPUT::

File read successfully!
Summary::
Correctly Classified Instances          42               93.3333 %
Incorrectly Classified Instances         3                6.6667 %
Kappa statistic                          0.8976
Mean absolute error                      0.0503
Root mean squared error                  0.1965
Relative absolute error                 11.1783 %
Root relative squared error             41.0994 %
Total Number of Instances               45     

===============================
Class wise detailed metrics::
                 TP Rate  FP Rate  Precision  Recall   F-Measure  MCC      ROC Area  PRC Area  Class
                 1.000    0.000    1.000      1.000    1.000      1.000    1.000     1.000     Iris-setosa
                 0.923    0.063    0.857      0.923    0.889      0.843    0.978     0.950     Iris-versicolor
                 0.900    0.040    0.947      0.900    0.923      0.865    0.982     0.980     Iris-virginica
Weighted Avg.    0.933    0.036    0.935      0.933    0.934      0.895    0.986     0.977     

===============================
Confusion Matrix
  a  b  c   <-- classified as
 12  0  0 |  a = Iris-setosa
  0 12  1 |  b = Iris-versicolor
  0  2 18 |  c = Iris-virginica
