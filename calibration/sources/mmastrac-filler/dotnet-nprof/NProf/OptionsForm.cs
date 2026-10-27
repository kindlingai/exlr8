using System;
using System.IO;
using System.Drawing;
using System.Collections;
using System.ComponentModel;
using System.Windows.Forms;
using Genghis.Windows.Forms;
using System.Collections.Generic;

namespace NProf
{
	public class PropertiesForm : System.Windows.Forms.Form
	{
		private System.Windows.Forms.Label label1;
		private System.Windows.Forms.Label label3;
		private System.Windows.Forms.CheckBox debugProfiler;
		private System.Windows.Forms.TextBox application;
		private System.Windows.Forms.TextBox arguments;
		private System.Windows.Forms.Button _btnCreateProject;
		private System.Windows.Forms.Button _btnCancel;
		private System.Windows.Forms.ToolTip _ttToolTips;
		private System.Windows.Forms.Button _btnBrowseApplication;
		private System.ComponentModel.IContainer components;

		private ProjectInfo project;
		private RadioButton radioButton1;
		private RadioButton radioButton2;
		private FlowLayoutPanel recentProjects;
		private ProfilerProjectMode projectMode;

		public PropertiesForm(ProfilerProjectMode mode)
		{
			//
			// Required for Windows Form Designer support
			//
			InitializeComponent();

			//
			// TODO: Add any constructor code after InitializeComponent call
			//
			projectMode = ProfilerProjectMode.CreateProject;

			project = new ProjectInfo( ProjectType.File );
			this.Mode = mode;
			if (mode == ProfilerProjectMode.CreateProject)
			{
				List<string> files=SerializationHandler.GetRecentlyUsed();
				foreach (string file in files.GetRange(0,Math.Min(files.Count,5)))
				{
					LinkLabel label=new LinkLabel();
					label.AutoSize = true;
					label.Text=file;
					label.Click += delegate
					{
						ProfilerForm.form.Project = SerializationHandler.OpenProjectInfo(label.Text);
						Close();
					};
					recentProjects.Controls.Add(label);

				}
			}
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
			this.components = new System.ComponentModel.Container();
			this.debugProfiler = new System.Windows.Forms.CheckBox();
			this._btnBrowseApplication = new System.Windows.Forms.Button();
			this.application = new System.Windows.Forms.TextBox();
			this.label1 = new System.Windows.Forms.Label();
			this.arguments = new System.Windows.Forms.TextBox();
			this.label3 = new System.Windows.Forms.Label();
			this._btnCreateProject = new System.Windows.Forms.Button();
			this._btnCancel = new System.Windows.Forms.Button();
			this._ttToolTips = new System.Windows.Forms.ToolTip(this.components);
			this.radioButton1 = new System.Windows.Forms.RadioButton();
			this.radioButton2 = new System.Windows.Forms.RadioButton();
			this.recentProjects = new System.Windows.Forms.FlowLayoutPanel();
			this.SuspendLayout();
			// 
			// debugProfiler
			// 
			this.debugProfiler.FlatStyle = System.Windows.Forms.FlatStyle.System;
			this.debugProfiler.Location = new System.Drawing.Point(307, 62);
			this.debugProfiler.Name = "debugProfiler";
			this.debugProfiler.Size = new System.Drawing.Size(128, 24);
			this.debugProfiler.TabIndex = 6;
			this.debugProfiler.Text = "Debug profiler hook";
			this._ttToolTips.SetToolTip(this.debugProfiler, "Launch the debugger as soon as the profilee starts");
			// 
			// _btnBrowseApplication
			// 
			this._btnBrowseApplication.FlatStyle = System.Windows.Forms.FlatStyle.System;
			this._btnBrowseApplication.Location = new System.Drawing.Point(437, 12);
			this._btnBrowseApplication.Name = "_btnBrowseApplication";
			this._btnBrowseApplication.Size = new System.Drawing.Size(75, 23);
			this._btnBrowseApplication.TabIndex = 3;
			this._btnBrowseApplication.Text = "Browse...";
			this._btnBrowseApplication.Click += new System.EventHandler(this._btnBrowseApplication_Click);
			// 
			// application
			// 
			this.application.Location = new System.Drawing.Point(119, 12);
			this.application.Name = "application";
			this.application.Size = new System.Drawing.Size(312, 20);
			this.application.TabIndex = 0;
			this._ttToolTips.SetToolTip(this.application, "Locate the execute to profile");
			// 
			// label1
			// 
			this.label1.Location = new System.Drawing.Point(15, 12);
			this.label1.Name = "label1";
			this.label1.Size = new System.Drawing.Size(100, 23);
			this.label1.TabIndex = 3;
			this.label1.Text = "Application to run:";
			// 
			// arguments
			// 
			this.arguments.Location = new System.Drawing.Point(119, 36);
			this.arguments.Name = "arguments";
			this.arguments.Size = new System.Drawing.Size(312, 20);
			this.arguments.TabIndex = 1;
			this._ttToolTips.SetToolTip(this.arguments, "Enter any command line arguments to pass to the above executable");
			// 
			// label3
			// 
			this.label3.Location = new System.Drawing.Point(15, 36);
			this.label3.Name = "label3";
			this.label3.Size = new System.Drawing.Size(100, 23);
			this.label3.TabIndex = 6;
			this.label3.Text = "Arguments:";
			// 
			// _btnCreateProject
			// 
			this._btnCreateProject.DialogResult = System.Windows.Forms.DialogResult.OK;
			this._btnCreateProject.FlatStyle = System.Windows.Forms.FlatStyle.System;
			this._btnCreateProject.Location = new System.Drawing.Point(307, 227);
			this._btnCreateProject.Name = "_btnCreateProject";
			this._btnCreateProject.Size = new System.Drawing.Size(96, 24);
			this._btnCreateProject.TabIndex = 2;
			this._btnCreateProject.Text = "OK";
			this._btnCreateProject.Click += new System.EventHandler(this._btnCreateProject_Click);
			// 
			// _btnCancel
			// 
			this._btnCancel.DialogResult = System.Windows.Forms.DialogResult.Cancel;
			this._btnCancel.FlatStyle = System.Windows.Forms.FlatStyle.System;
			this._btnCancel.Location = new System.Drawing.Point(416, 227);
			this._btnCancel.Name = "_btnCancel";
			this._btnCancel.Size = new System.Drawing.Size(96, 24);
			this._btnCancel.TabIndex = 3;
			this._btnCancel.Text = "Cancel";
			// 
			// radioButton1
			// 
			this.radioButton1.AutoSize = true;
			this.radioButton1.Checked = true;
			this.radioButton1.Location = new System.Drawing.Point(18, 69);
			this.radioButton1.Name = "radioButton1";
			this.radioButton1.Size = new System.Drawing.Size(68, 17);
			this.radioButton1.TabIndex = 7;
			this.radioButton1.TabStop = true;
			this.radioButton1.Text = "Sampling";
			this.radioButton1.UseVisualStyleBackColor = true;
			// 
			// radioButton2
			// 
			this.radioButton2.AutoSize = true;
			this.radioButton2.Location = new System.Drawing.Point(18, 93);
			this.radioButton2.Name = "radioButton2";
			this.radioButton2.Size = new System.Drawing.Size(97, 17);
			this.radioButton2.TabIndex = 8;
			this.radioButton2.Text = "Instrumentation";
			this.radioButton2.UseVisualStyleBackColor = true;
			// 
			// recentProjects
			// 
			this.recentProjects.Location = new System.Drawing.Point(17, 128);
			this.recentProjects.Name = "recentProjects";
			this.recentProjects.Size = new System.Drawing.Size(469, 93);
			this.recentProjects.TabIndex = 9;
			// 
			// PropertiesForm
			// 
			this.AcceptButton = this._btnCreateProject;
			this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
			this.CancelButton = this._btnCancel;
			this.ClientSize = new System.Drawing.Size(524, 263);
			this.Controls.Add(this.recentProjects);
			this.Controls.Add(this.radioButton2);
			this.Controls.Add(this.radioButton1);
			this.Controls.Add(this.label3);
			this.Controls.Add(this.debugProfiler);
			this.Controls.Add(this.arguments);
			this.Controls.Add(this.label1);
			this.Controls.Add(this._btnCancel);
			this.Controls.Add(this.application);
			this.Controls.Add(this._btnCreateProject);
			this.Controls.Add(this._btnBrowseApplication);
			this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedDialog;
			this.MaximizeBox = false;
			this.MinimizeBox = false;
			this.Name = "PropertiesForm";
			this.ShowInTaskbar = false;
			this.Text = "Create Profiler Project";
			this.Load += new System.EventHandler(this.ProfilerProjectOptionsForm_Load);
			this.ResumeLayout(false);
			this.PerformLayout();

		}
		#endregion

		public ProfilerProjectMode Mode
		{
			get { return projectMode; }
			set { projectMode = value; }
		}

		public ProjectInfo Project
		{
			get { return project; }
			set { project = value; }
		}

		private void _btnBrowseApplication_Click(object sender, System.EventArgs e)
		{
			OpenFileDialog ofd = new OpenFileDialog();
			ofd.Filter = "Executable files (*.exe)|*.exe";
			DialogResult dr = ofd.ShowDialog();
			if ( dr == DialogResult.OK )
			{
				application.Text = ofd.FileName;
				application.Focus();
				application.SelectAll();
			}
		}

		private void ProfilerProjectOptionsForm_Load(object sender, System.EventArgs e)
		{
			if ( projectMode == ProfilerProjectMode.CreateProject )
			{
				this.Text = "Create Profiler Project";
				_btnCreateProject.Text = "Create Project";
			}
			else
			{
				this.Text = "Modify Profiler Project Options";
				_btnCreateProject.Text = "OK";
			}

			application.Text = project.ApplicationName;
			arguments.Text = project.Arguments;
			debugProfiler.Checked = project.DebugProfiler;
		}

		private void _btnCreateProject_Click(object sender, System.EventArgs e)
		{
			project.ApplicationName = application.Text;
			project.Arguments = arguments.Text;
			project.DebugProfiler = debugProfiler.Checked;
		}

		public enum ProfilerProjectMode
		{
			CreateProject,
			ModifyProject,
		}
	}
}
