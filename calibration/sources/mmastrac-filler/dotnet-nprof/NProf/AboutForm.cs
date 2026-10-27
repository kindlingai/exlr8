/***************************************************************************
                          profiler.cpp  -  description
                             -------------------
    begin                : Sat Jan 18 2003
    copyright            : (C) 2003,2004,2005,2006 by Matthew Mastracci, Christian Staudenmeyer
    email                : mmastrac@canada.com
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
using System;
using System.IO;
using System.Drawing;
using System.Collections;
using System.ComponentModel;
using System.Windows.Forms;

namespace NProf
{
	/// <summary>
	/// Summary description for AboutForm.
	/// </summary>
	public class AboutForm : System.Windows.Forms.Form
	{
		private System.Windows.Forms.Label label1;
		private System.Windows.Forms.RichTextBox richTextBox1;
		private System.Windows.Forms.Button button1;
		private System.Windows.Forms.Label label3;
		private System.Windows.Forms.LinkLabel linkLabel1;
		private System.Windows.Forms.LinkLabel linkLabel2;
		/// <summary>
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.Container components = null;

		public AboutForm()
		{
			//
			// Required for Windows Form Designer support
			//
			InitializeComponent();

			//
			// TODO: Add any constructor code after InitializeComponent call
			//
		}

		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		protected override void Dispose( bool disposing )
		{
			if( disposing )
			{
				if(components != null)
				{
					components.Dispose();
				}
			}
			base.Dispose( disposing );
		}

		#region Windows Form Designer generated code
		/// <summary>
		/// Required method for Designer support - do not modify
		/// the contents of this method with the code editor.
		/// </summary>
		private void InitializeComponent()
		{
			this.label1 = new System.Windows.Forms.Label();
			this.richTextBox1 = new System.Windows.Forms.RichTextBox();
			this.button1 = new System.Windows.Forms.Button();
			this.label3 = new System.Windows.Forms.Label();
			this.linkLabel1 = new System.Windows.Forms.LinkLabel();
			this.linkLabel2 = new System.Windows.Forms.LinkLabel();
			this.SuspendLayout();
			// 
			// label1
			// 
			this.label1.Location = new System.Drawing.Point(8, 8);
			this.label1.Name = "label1";
			this.label1.Size = new System.Drawing.Size(392, 24);
			this.label1.TabIndex = 0;
			this.label1.Text = "nprof is Copyright 2002-2004 by Matthew Mastracci.  All Rights Reserved.";
			// 
			// richTextBox1
			// 
			this.richTextBox1.Font = new System.Drawing.Font("Courier New", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((System.Byte)(0)));
			this.richTextBox1.Location = new System.Drawing.Point(8, 40);
			this.richTextBox1.Name = "richTextBox1";
			this.richTextBox1.ReadOnly = true;
			this.richTextBox1.Size = new System.Drawing.Size(592, 240);
			this.richTextBox1.TabIndex = 1;
			this.richTextBox1.Text = "01234567890123456789012345678901234567890123456789012345678901234567890123456789";
			this.richTextBox1.WordWrap = false;
			// 
			// button1
			// 
			this.button1.DialogResult = System.Windows.Forms.DialogResult.OK;
			this.button1.FlatStyle = System.Windows.Forms.FlatStyle.System;
			this.button1.Location = new System.Drawing.Point(268, 368);
			this.button1.Name = "button1";
			this.button1.TabIndex = 2;
			this.button1.Text = "OK";
			// 
			// label3
			// 
			this.label3.Location = new System.Drawing.Point(8, 328);
			this.label3.Name = "label3";
			this.label3.Size = new System.Drawing.Size(592, 32);
			this.label3.TabIndex = 6;
			this.label3.Text = "In addition to the terms of the GPL above, permission is granted to link nprof ag" +
				"ainst the Magic Library and the Genghis Library.";
			// 
			// linkLabel1
			// 
			this.linkLabel1.Location = new System.Drawing.Point(456, 8);
			this.linkLabel1.Name = "linkLabel1";
			this.linkLabel1.Size = new System.Drawing.Size(144, 16);
			this.linkLabel1.TabIndex = 7;
			this.linkLabel1.TabStop = true;
			this.linkLabel1.Text = "http://nprof.sourceforge.net";
			this.linkLabel1.LinkClicked += new System.Windows.Forms.LinkLabelLinkClickedEventHandler(this.linkLabel_LinkClicked);
			// 
			// linkLabel2
			// 
			this.linkLabel2.LinkArea = new System.Windows.Forms.LinkArea(48, 28);
			this.linkLabel2.Location = new System.Drawing.Point(8, 296);
			this.linkLabel2.Name = "linkLabel2";
			this.linkLabel2.Size = new System.Drawing.Size(592, 16);
			this.linkLabel2.TabIndex = 8;
			this.linkLabel2.TabStop = true;
			this.linkLabel2.Text = "Portions copyright 2002-2003 The Genghis Group (http://www.genghisgroup.com/).";
			this.linkLabel2.LinkClicked += new System.Windows.Forms.LinkLabelLinkClickedEventHandler(this.linkLabel_LinkClicked);
			// 
			// AboutForm
			// 
			this.AcceptButton = this.button1;
			this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
			this.CancelButton = this.button1;
			this.ClientSize = new System.Drawing.Size(610, 399);
			this.Controls.Add(this.linkLabel2);
			this.Controls.Add(this.linkLabel1);
			this.Controls.Add(this.label3);
			this.Controls.Add(this.button1);
			this.Controls.Add(this.richTextBox1);
			this.Controls.Add(this.label1);
			this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedDialog;
			this.MaximizeBox = false;
			this.MinimizeBox = false;
			this.Name = "AboutForm";
			this.ShowInTaskbar = false;
			this.Text = "About nprof";
			this.Load += new System.EventHandler(this.AboutForm_Load);
			this.ResumeLayout(false);

		}
		#endregion

		private void AboutForm_Load(object sender, System.EventArgs e)
		{
			using (StreamReader reader = new System.IO.StreamReader(System.Reflection.Assembly.GetExecutingAssembly().GetManifestResourceStream("NProf.Resources.GPL.txt")))
			{
				richTextBox1.Text = reader.ReadToEnd();
			}
		}

		private void linkLabel_LinkClicked(object sender, System.Windows.Forms.LinkLabelLinkClickedEventArgs e)
		{
			LinkLabel label = (LinkLabel)sender;

			string url = label.Text.Substring(e.Link.Start, e.Link.Length);

			System.Diagnostics.Process.Start(url);
		}
	}
}
