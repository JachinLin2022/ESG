from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoModelForQuestionAnswering
from transformers import TrainingArguments
from transformers import Trainer
import torch
from datasets import *
import re
import collections
import numpy as np
import heapq
import json

input = '''
2021
Environmental, Social
and Governance Report
Stock Code: 0008
CONTENTS
Group Managing Director’s Message 4
About This Report 5
Highlights 8
1. Corporate Social Responsibility at PCCW 9
1.1 CSR Governance Structure and
Oversight Responsibilities
9
1.2 ESG Strategy 11
1.3 Ethics and Integrity 12
1.4 Stakeholder Engagement and
Materiality Review
14
1.5 External Recognition 17
2. Our People 18
2.1 Employee Well-being 19
2.2 Occupational Safety and Health 21
2.3 Talent Retention and Development 23
2.4 Diversity and Inclusiveness 26
3. Our Environment 28
3.1 Climate Change, Energy Consumption
and GHG Emissions
29
3.2 Sustainable Use of Resources 31
3.3 Environmental Targets 34
3.4 Building a Smart City 35
3.5 Employee Environmental Awareness 36
4. Our Community 37
4.1 Community Engagement 38
4.2 Digital Empowerment 46
5. Our Customers 47
5.1 Customer Data Privacy and Security 47
5.2 Reliable and Responsible Services and
Products
49
5.3 Content Dissemination and
Responsible Advertising
54
5.4 Customer Service and Satisfaction 54
6. Our Supply Chain Management 56
6.1 Supplier Code of Conduct 56
6.2 Supplier Selection and Monitoring 57
6.3 Sustainable Procurement 58
Assurance Report 59
External Charters and Membership 60
Performance Data Summary 61
References to HKEX ESG Reporting Guide 63
GROUP MANAGING DIRECTOR’S MESSAGE
GROUP MANAGING DIRECTOR’S MESSAGE
PCCW Environmental, Social and Governance Report 2021
04
In terms of the environment, we are pleased to have
established a set of Group-wide environmental targets
during the year, modeled after our industry peers and
international goals. These will guide enhancements to
our energy efficiency as well as the reduction of our
waste generation, electricity and water consumption, and
greenhouse gas (“GHG”) emission.
For our society, while the new normal has brought out issues
such as the digital divide, it has also presented a clearer
picture of where exactly our expertise can most effectively
contribute, such as offering smartphones and training for
the elderly to make use of anti-pandemic mobile apps.
Internally, we strived to foster a safe and healthy working
environment by offering time off for staff after each dose
of COVID-19 vaccination. In reviewing our promotion of
employee health and well-being, we implemented seminars
and workshops on healthy eating as well as physical and
mental wellness to better equip our colleagues in the battle
against the coronavirus.
With maturing strategies and unwavering efforts, we
shall further identify opportunities for improvement in our
operations for greater sustainability. We would also like to
thank all of our stakeholders who participated and provided
valuable comments in this year’s stakeholder engagement
exercise. Please continue to lend us your support for our
ever-evolving sustainability endeavors.
BG Srinivas
Group Managing Director
February 24, 2022
I am happy to present PCCW’s Environmental, Social and
Governance (“ESG”) Report for 2021 and share our updates
on sustainability.
Rooted in communications and inspired by technology, both
the history and prospects of PCCW are entwined with the
everyday lives of people from all walks of life. That is why we
have long committed ourselves to advancing the interests of
our planet and society, through means such as upholding
the strictest standards for all aspects of our operations.
In today’s world, where life is hinged onto networks and
hampered by social distancing, we are humbly reminded
of our mission – to empower, connect and transform. To
achieve this, we must not only continue our ESG efforts but
refine them in a way that meets the ever-rising expectations
of our stakeholders, the ever-growing needs of our business,
and ever-more stringent global regulations.
PCCW Environmental, Social and Governance Report 2021
ABOUT THIS REPORT
ABOUT THIS REPORT
05
This is the ESG Report for PCCW Limited (“PCCW” or the “Company”) and its subsidiaries (collectively referred to as the
“Group” in this report).
PCCW Limited (SEHK: 0008) is a global company headquartered in Hong Kong which holds interests in telecommunications,
media, IT solutions, property development and investment, and other businesses.
The Company holds a majority stake in the HKT Trust and HKT Limited (“HKT”, SEHK: 6823), Hong Kong’s premier
telecommunications service provider and leading operator of fixed-line, broadband, mobile communication and media
entertainment services. HKT delivers end-to-end integrated solutions employing emerging technologies to assist enterprises
in transforming their businesses. HKT has also built a digital ecosystem integrating its loyalty program, e-Commerce, travel,
insurance, Big Data Analytics, FinTech and HealthTech services to deepen its relationship with customers.
PCCW owns a fully integrated multimedia and entertainment group in Hong Kong engaged in the provision of over-the-top
(“OTT”) video service locally and in other places in the region.
Through HK Television Entertainment Company Limited, PCCW also operates a domestic free TV service in Hong Kong.
Also wholly owned by the Group, PCCW Solutions is a leading IT and business process outsourcing provider in Hong Kong,
mainland China and Southeast Asia.
In addition, PCCW holds a stake in Pacific Century Premium Developments Limited (“PCPD”, SEHK: 0432) and other overseas
investments.
PCCW has created a variety of well-known products and service brands. Some of the more recognizable brands are shown as
follows:
Property
Development
Telecommunications Media IT Solutions
PCCW Environmental, Social and Governance Report 2021
ABOUT THIS REPORT
06
Reporting Scope
This report covers PCCW’s ESG accomplishments and challenges from January 1 to December 31, 2021, as well as our
ongoing initiatives to enhance our ESG performance. It encompasses PCCW’s core businesses based in Hong Kong, namely
telecommunications, media and IT solutions, respectively operated through our key subsidiaries HKT, PCCW Media and
PCCW Solutions, including the operations of our offices, retail shops, data centers, exchange sites, telecommunications sites
and transmissions, unless otherwise specified. Where relevant, it also references the activities of subsidiaries and outsourced
operations. In 2021, our businesses in Hong Kong accounted for 77.6% of the total revenue of PCCW.
This report does not cover PCCW’s joint ventures and PCPD. PCPD issues a separate sustainability report.
The qualitative and quantitative information regarding PCCW’s approach, initiatives and priorities in managing material ESG
aspects are disclosed in the report. For further disclosures on corporate governance, please refer to the Corporate Governance
Report of the PCCW Annual Report 2021.
Reporting Standards and External Assurance
This report has been prepared in accordance with provisions of the Environmental, Social and Governance Reporting Guide
(the “ESG Reporting Guide”) in Appendix 27 to the Rules Governing the Listing of Securities on The Stock Exchange of Hong
Kong Limited (“HKEX”).
This report serves as an important channel to connect and communicate with our stakeholders. We believe that ensuring the
relevance and importance of our ESG information is critical to our stakeholders. As such, the report has been prepared and
presented with reference to the Reporting Principles set out in the ESG Reporting Guide.
Materiality Quantitative
Materiality was assessed based on the results obtained from
stakeholder engagement. The threshold for sustainability
topics to become material was reviewed and confirmed by
the top management to ensure that they were sufficiently
important to our stakeholders.
A cloud-based data management platform was deployed to
collect our ESG metrics, keep track of our performance and
assist on target setting.
Where applicable, we compared year-to-year data
and discussed their implications. In the reporting of
emissions and energy consumption, relevant standards,
methodologies, assumptions, and conversion factors have
been disclosed.
Balance Consistency
The content and data provided in the report are unbiased.
We discussed both our achievements and room for
improvement in all ESG aspects.
This report adopts consistent methodologies to allow a
fair comparison of our performance over time. Where
applicable, we disclosed the changes to the methods or key
performance indicators (“KPIs”) used.
PCCW Environmental, Social and Governance Report 2021
ABOUT THIS REPORT
07
The environmental and social performance data in the report have been independently reviewed and verified by the Hong
Kong Quality Assurance Agency (“HKQAA”). Please refer to the Assurance Report on page 59 for the verification scope and
conclusion.
The Board of Directors (the “Board”) of PCCW is accountable for our ESG strategies and reporting, as well as responsible for
overseeing and managing our ESG-related risks. This report has been reviewed and approved by the Board of PCCW.
Available in both Chinese and English, this report can be accessed on HKEX’s website and PCCW’s website.
We value stakeholders’ views and suggestions. Please share your feedback on our ESG management and reporting with our
Group Communications department.
Mail: 41/F, PCCW Tower, Taikoo Place, 979 King’s Road, Quarry Bay, Hong Kong
Phone: +852 2888 2888
Fax: +852 2962 5634
Email: esg@pccw.com
HIGHLIGHTS
HIGHLIGHTS
PCCW Environmental, Social and Governance Report 2021
08
Donated smartphones
with data plan and broadband service
to the elderly and low-income families
>28% of employees aged below 30
>30% of senior management roles held
by female staff
Over 300 sessions
of health and safety-related training
1:1.41
female to male staff
Diversity Occupational Safety and Health
Health and Well-being
Emissions
Environmental Targets
Set out reduction targets for electricity consumption, GHG emissions, water consumption and general
waste, to cover the period up to 2025.
Electricity consumption
decreased more than
12.9 GWh
Total GHG emissions
intensity per employee
decreased by 11%
Issued a new Information Technology Security Policy
to enhance company information and customer
data protection capabilities
Upgraded 15
vehicles to Euro 6 or
electric cars
80% of our suppliers are Hong Kong-based
Governance Supply Chain
2 days of paid leave for each dose of
COVID-19 vaccination received
Over HK$20million
in monetary donations and in-kind sponsorships
Now TV awarded Best Corporate Social
Responsibility Media – Bronze in Sparks Awards
98%
of households covered by Fiber-to-the-Home
(“FTTH”) and Wireless-to-the-Home (“WTTH”)
100%
mobile reliability
99.99%
broadband network stability
Responsible Network Management Community Investment
48,845 customer compliments
ISO 22301 certied Business Continuity
Management Systems
Entered into multiple sustainability-linked loan
facilities, raising about US$1 billion since 2020
Customer Satisfaction Sustainability-linked Loan
Business Continuity
PCCW Environmental, Social and Governance Report 2021
1. CORPORATE SOCIAL RESPONSIBILITY AT PCCW
CORPORATE SOCIAL RESPONSIBILITY AT PCCW
09
1.1 CSR Governance Structure and Oversight Responsibilities
PCCW Board of Directors
Executive
Committee
Remuneration
Committee
Nomination
Committee
Audit
Committee
Regulatory
Compliance Committee
Operational
Committee
Risk Management, Controls and
Compliance Committee
Corporate Social Responsibility
Committee
Environmental
Advisory Group
Departments and
Business Units
Group Risk Management
and Compliance
Departmental Corporate Social
Responsibility Representatives
Corporate Social
Responsibility Team
Led by our top management, PCCW is committed to integrating corporate social responsibility (“CSR”) into its business
operation. The Board, assisted by various committees, formulates strategies and maintains oversight of our ESG performance.
The Executive Committee oversees several sub-committees and working groups to ensure its CSR Policy and risk management
systems are implemented effectively.
Roles and responsibilities in managing CSR matters are defined within PCCW as follows:
Overseen by top management
Board of Directors • Monitors corporate governance practices and procedures
• Maintains appropriate and effective risk management and internal control systems of the
Group to ensure compliance with applicable rules and regulations
• Approves CSR Policy and Corporate Responsibility (“CR”) Policy
• Reviews and approves the ESG Report
Executive Committee • Operates as a general management committee with overall delegated authority from the
Board
PCCW Environmental, Social and Governance Report 2021
CORPORATE SOCIAL RESPONSIBILITY AT PCCW
10
Audit Committee • Assists the Board in ensuring the objectivity and credibility of financial reporting, and that
the directors have exercised care, diligence and skills prescribed by law when presenting
results to shareholders
• Assists the Board in ensuring that effective risk management and internal control systems
are in place and good corporate governance standards and practices are maintained
• Reviews and recommends the ESG report for the Board’s approval
Risk Management,
Controls and
Compliance Committee
• Reviews procedures for preparation of PCCW annual and interim reports and, from time
to time, corporate policies of the Group to ensure compliance with the various rules and
obligations of a Hong Kong-listed company
• Assists the Board and/or the Audit Committee in the review of the effectiveness of the
risk management and internal control systems of the Group on an ongoing basis
• Reviews and recommends the ESG Report for submission to the Audit Committee
Managing CSR issues and implementing CSR initiatives
CSR Committee A sub-committee reporting to the Executive Committee, chaired by the Head of Group
Communications and comprising Group department and unit heads, which:
• Reviews the Company’s CSR strategy, principles and policies to ensure the Company
operates in a manner that enhances its positive contribution to society and the
environment
• Oversees and provides guidance and direction for CSR practices and procedures
• Monitors progress on CSR-related initiatives
• Monitors progress towards the relevant goals and targets
• Reviews the ESG Report
Environmental Advisory
Group
An internal advisory body comprising Group unit heads that:
• Advises on environmental policies and targets and makes recommendations to the CSR
Committee
• Assists in the coordination of Business Units (“BU”) and cross-BU environmental initiatives
CSR Team Together with departmental CSR representatives, it:
• Promotes CSR internally and externally
• Organizes and implements CSR initiatives
• Prepares the ESG Report
Departmental CSR
Representatives
A total of 20 departmental CSR representatives:
• Serve as a bridge between the CSR Team and the departments/BUs
• Facilitate implementation of CSR initiatives
• Raise CSR awareness among colleagues
• Assist in ESG reporting and ESG-related surveys
Departments and BUs • Implement CSR practices and ensure CSR compliance in daily operations
PCCW Environmental, Social and Governance Report 2021
CORPORATE SOCIAL RESPONSIBILITY AT PCCW
11
The Group’s Enterprise Risk Management framework is adopted with reference to ISO 31000:2018 Risk Management –
Guidelines. Through the “Three Lines of Defence” operating model, our Board of Directors defines as well as regularly
evaluates and determines significant risks that may impact the Group’s performance.
Our Corporate Incident Response Plan ensures business continuity with minimum interruption to our operations. As stipulated
in the Corporate Incident Response Plan, the Corporate Incident Response Team provides leadership, strategic direction,
communication and consistent response in dealing with activities arising from corporate incidents.
Group Risk Management and Compliance is responsible for the supervision of enterprise risk management activities while
reviewing significant aspects of risk exposures of the Group through reporting to the Audit Committee at each regularly
scheduled meeting, including key risks of the Group and the appropriate mitigation and/or transfer of identified risks. The
operating units of the Group, as risk owners, identify, evaluate, mitigate and monitor their own risks, and report such
risk management activities to Group Risk Management and Compliance on a regular basis. Group Risk Management and
Compliance assesses and presents regular reports to the Risk Management, Controls and Compliance Committee at each
regularly scheduled meeting.
Group Internal Audit maintains primary accountability to the Board and independence from the responsibilities of
management.
For more details on the composition and responsibilities of various committees of the Board, our risk management and
internal controls, as well as the principal risks and uncertainties identified in relation to our key areas of operations, please
refer to the Corporate Governance Report and the Report of the Directors in PCCW’s Annual Report 2021.
1.2 ESG Strategy
As a leading provider of telecommunications, media entertainment and enterprise IT solutions locally and globally, PCCW
upholds its responsibilities to promote sustainability and make meaningful contributions to society.
Our management approach is to run our business in an ethically, socially and environmentally responsible manner, supporting
and connecting the communities we serve. This shall be achieved alongside service excellence and financial returns.
Our CSR Policy sets forth our overarching principles, objectives and approach in key areas of CSR management with reference
to ISO 26000 (Guidance on social responsibility). The policy is applicable to all directors, officers and employees of the Group,
and communicated with third parties such as suppliers and contractors, where applicable. We regularly review our CSRrelated policies and update them upon the Board’s approval.
PCCW Environmental, Social and Governance Report 2021
CORPORATE SOCIAL RESPONSIBILITY AT PCCW
12
Detailed policies, guidelines and procedures are in place to guide our operating practices across departments.
Community objectives
in CSR Policy
Community
Our CSR Management
Approach
Environment
Employment and
Labor
Supply Chain
Management
Customers and
Marketplace
CSR Policy
CR Policy
Anti-Bribery and Corruption Policy
Anti-Money Laundering and Counter-Terrorist Financing Policy
Information Technology Security Policy
Energy and Water
Management Policy and Guidelines
Gas Emission Reduction Policy
Human Resources Policy Manual
Employee Handbook
Occupational Safety and Health Policy
Supplier Code of Conduct
Group Purchasing Policy and Principles
Personal Data Privacy Policy
Privacy Statement
Intellectual Property Rights Policy
Fraud and Security Incident
Management Policy
Sensitive Information Monitoring
Policy
1.3 Ethics and Integrity
PCCW is committed to conducting its business and operations with high standards of ethics, honesty and integrity in
accordance with all applicable laws and regulations and the Group’s policies. This requires all members of the Group to
uphold an aligned standard of behavior that exceeds statutory mandates.
In this regard, our CR Policy and other PCCW Group policies provide practical guidelines on business conduct. Applicable to
our directors, officers and employees, these policies ensure responsible behavior and protection for stakeholders’ rights in
case of breach. Topics covered include but are not limited to:
• Bribery, gifts and entertainment
• Conflicts of interest
• Considerate and civic responsibility
• Discrimination, harassment and inappropriate conduct
• Equal opportunities
• Fair competition
• Inside information
• Money laundering and terrorist financing
• Privacy and information protection
• Whistleblowing to report improper conduct
• Workplace health and safety
PCCW Environmental, Social and Governance Report 2021
CORPORATE SOCIAL RESPONSIBILITY AT PCCW
13
Anti-bribery and corruption
PCCW expects all directors, officers, employees and external parties acting in any capacity on behalf of PCCW to closely
adhere to our Anti-Bribery and Corruption (“ABC”) Policy. We do not tolerate any form of bribery or corruption at any level.
The ABC Policy outlines the roles and responsibilities of employees and controls implemented by each BU to reduce the risk
of corruption, and ensures compliance with Group standards as well as all relevant laws and regulations. In addition, offering
or promising to give or accept any gift or hospitality to reward or retain a business, or authorizing any bribery or corruption in
any business dealings that involve our Group and government officials, our customers, vendors or employees are prohibited.
Besides putting in place the ABC Policy, a corresponding Procedures Manual has been established to provide guidance on the
mitigation of potential bribery and corruption risks. The Policy and Procedures Manual are based on the underlying principles
of UK Bribery Act 2010 (“UKBA”), which is internationally recognized as one of the highest ABC standards.
Anti-money laundering and counter-terrorist financing
The Group has also issued a new Anti-Money Laundering (“AML”) and Counter-Terrorist Financing (“CTF”) Policy. The Group is
committed to providing an effective and systematic Group-wide AML and CTF framework for all of its subsidiaries, employees
and associated parties to follow in their business dealings and daily operations. Regular risk assessments and monitoring work
on each BU level are undertaken to assure strict compliance with all applicable laws and regulations.
When potential conflicts of interest arise, employees may consult their line manager for clarification. Application and
declaration forms are available on the Company’s intranet.
Whistleblowing
PCCW’s Whistleblower Policy and Whistleblower Procedures Manual foster a positive culture for all internal and external
stakeholders to report actual or suspected improper conduct in confidence to the Audit Committee via the Group Internal
Audit function. Employees can submit written reports by mail or via the secure and confidential email address and hotline
managed independently by the Group Internal Audit function.
All whistleblowing cases are treated in strict accordance with the procedures set out in the Group’s Whistleblower Policy and
Whistleblower Procedures Manual, which are reviewed periodically. An independent and appropriate senior member of the
Company’s staff is appointed to act on behalf of the Chairman of the Audit Committee, such as the Head of Group Internal
Audit, as the case manager with responsibility for the conduct, management, and reporting of the matter. Upon completion
of the investigation, a report, including its final disposition, the impact, implications and recommendation for improvement, as
applicable, is provided to the Risk Management, Controls and Compliance Committee of the Company for consideration and
for further reporting to the Audit Committee of the Company as they deem appropriate.
Ensuring compliance
Any individual who violates PCCW policies, procedures and guidelines may receive verbal or written warnings or be summarily
dismissed depending on the severity of the infraction. We monitor and identify applicable laws and regulations which have
a significant impact on the Group as well as its latest development. Various measures including internal controls, approval
procedures and training are in place to raise staff awareness of the Company’s ethics and integrity standards. For more
details, please refer to the Report of the Directors in PCCW’s Annual Report 2021.
PCCW Environmental, Social and Governance Report 2021
CORPORATE SOCIAL RESPONSIBILITY AT PCCW
14
Employees must sign a declaration of acknowledgement and compliance with our CR Policy upon employment and also
during their annual performance review. All new on-boarding employees are required to complete induction training on risk
and compliance which covers the topics of enterprise risk management and compliance, ABC, AML and CTF, data privacy,
technology risk and cybersecurity, international trade compliance and whistleblower protection. In addition, there is a
separate training module on fraud awareness as well.
Advocating the importance of compliance culture and values, Group Risk Management and Compliance offers related
training to all operating units on an ongoing basis.
In 2021, there were no cases of non-compliance with the Prevention of Bribery Ordinance (Cap. 201) and other applicable
laws and regulations related to corruption at PCCW, nor any legal cases concerned with corrupt practices brought against the
Group or its employees.
1.4 Stakeholder Engagement and Materiality Review
Through regular stakeholder engagement, PCCW aims to make stakeholder-inclusive decisions and review our management
priorities and performance. We also disclose material information in response to stakeholders’ needs and expectations. These
processes are guided by our CSR Policy, CR Policy and Shareholders Communication Policy.
Stakeholder groups
We recognize stakeholders’ rights to be heard and informed. Departments and BUs of PCCW maintain continuous
communication with stakeholders through various channels.
External stakeholders
Customers Shareholders, investors and
analysts
Community and media
• Service hotlines
• Website and social media
• Live webchat
• My HKT portal
• Customer satisfaction survey and
transaction survey
• Net promoter score survey
• Meetings
• Annual general meetings
• Annual, interim and ESG reports
• Circulars and press releases
• Analyst briefings
• Website of HKEX
• Campaigns
• Seminars
• Website and social media
• Press releases and conferences
• Media enquiries
Government and regulators Suppliers and business partners NGOs
• General liaison • Supplier review and assessment
visits
• Corporate volunteering
• Collaborative projects
Internal stakeholders – management and employees
• Face-to-face meetings
• Let’s Chat sessions
• Forums
• Town hall style gatherings
• Employee satisfaction survey
• Intranet
PCCW Environmental, Social and Governance Report 2021
CORPORATE SOCIAL RESPONSIBILITY AT PCCW
15
Materiality review
PCCW reviews the materiality of ESG-related topics based on stakeholder engagement activities on an annual basis. This year,
PCCW continued to review the list of ESG topics based on a peer benchmarking exercise and internal evaluation to identify
and prioritize topics that are material and relevant to the development of the industry and the Group.
Facilitated by an independent consultant, PCCW engaged both internal and external stakeholder groups through online
surveys, focus groups and interviews. Internally, we worked with the Group unit heads and staff members. Externally, we
engaged investors, suppliers and contractors, corporate clients, business partners, academia and community partners.
Through these in-depth conversations, we collected stakeholder feedback on PCCW’s ESG performance and received
suggestions on our future priorities. Participants were also invited to score ESG topics based on their importance to
stakeholders and PCCW’s business operation.
Our CSR Committee evaluated this feedback based on the analysis of the qualitative and quantitative input from the
stakeholder engagement exercises.
PCCW’s Materiality Matrix 2021
Importance to Stakeholders
Importance to Business Operation
1
3 2
4
6
5
11
7
23
13
17
20
16
15
22
8
14
21
19
9
18
12
25
26
10
24
PCCW Environmental, Social and Governance Report 2021
CORPORATE SOCIAL RESPONSIBILITY AT PCCW
16
ESG ASPECTS
Environmental
1 Energy efficiency
2 GHG emissions
3 Waste management
4 Climate change
5 Green ICT solutions
6 Employee environmental awareness
Employment and Labor Standards
7 Employee well-being*
8 Employee diversity
9 Employee retention and talent development
10 Occupational safety and health
11 Human rights
Supply Chain Management
12 Supply chain management
13 Sustainable procurement
Product Responsibility
14 Responsible advertising
15 Customer health and safety*
16 Reliable services and products
17 Customer service and satisfaction
18 Customer data privacy and protection
19 Information security and management
20 Content dissemination to different audience groups
21 Business innovation
Corporate Governance
22 Corporate governance and risk management
23 Anti-corruption
24 Competitive behavior
Community
25 Community investment
26 Technology and education initiatives
* ESG topics newly added in 2021
PCCW Environmental, Social and Governance Report 2021
CORPORATE SOCIAL RESPONSIBILITY AT PCCW
17
Based on the dialogues and the scores given to each ESG topic, PCCW noticed that top material aspects in 2021, located
in the upper-right corner of the matrix, generally fall under product responsibility and corporate governance. Customer
data privacy and protection remains as the top priority to stakeholders and the operations of PCCW, while the second most
material aspect is anti-corruption.
Please refer to the corresponding chapters in this report for details of PCCW’s policies, measures and responses relevant to
the material topics. In the future, we will continue to regularly communicate and engage with our stakeholders to promote
sustainable development.
1.5 External Recognition
In 2021, PCCW received an overall rating of A in the MSCI ESG rating update, ranking in the top 63% of global
telecommunication services peers.
PCCW is a constituent member of the Hang Seng Corporate Sustainability Index Series, Hang Seng ESG 50 Index and
FTSE4Good Index Series.
2. OUR PEOPLE
OUR PEOPLE
PCCW Environmental, Social and Governance Report 2021
18
Employees are among the company’s greatest assets. Through diverse talent strategies, PCCW strives to cultivate a pleasant,
inclusive and productive work environment for our 20,693 employees1
 globally. As an Employer of Choice, we support and
empower our employees to unleash their potential and succeed in their career.
Objectives
Cultivate a high performing and engaging
culture
Attract, develop and retain
the right talent with robust
bench planning and succession
Foster a vibrant and diverse
workforce providing the best
employee experience
Measures
Drive a total reward system that recognizes and
incentivizes good performance
Promote staff health and well-being for a healthy
workplace
Enable career mobility and development paths
across the Group
Enable cross-functional staff engagement and
connection
Number of employees in Hong Kong: • Number of employees outside Hong Kong:
• Full-time staff: 7,800 12,652
• Part-time staff (as full-time equivalent):
241
1 Excluding PCPD employees
Staff Profile
OUR PEOPLE
PCCW Environmental, Social and Governance Report 2021
19
We have in place comprehensive employment policies and procedures to ensure employees’ rights and benefits, and offer
competitive pay and career progression opportunities. These policies and procedures are formulated in accordance with all
relevant laws and regulations, which include:
• The four anti-discrimination ordinances (sex, disability, family status and race)
• Employees’ Compensation Ordinance (Cap. 282)
• Employment Ordinance (Cap. 57)
• Factories and Industrial Undertakings Ordinance (Cap. 59)
• Inland Revenue Ordinance (Cap. 112)
• Mandatory Provident Fund Schemes Ordinance (Cap. 485)
• Minimum Wage Ordinance (Cap. 608)
• Occupational Retirement Schemes Ordinance (Cap. 426)
• Occupational Safety and Health Ordinance (Cap. 509)
• Personal Data (Privacy) Ordinance (Cap. 486)
During the reporting period, there were no non-compliance cases regarding the relevant laws and regulations.
2.1 Employee Well-being
We promote work-life balance and implement measures to maintain the health and well-being of our employees.
Family friendliness
We offer flexible work arrangements to support employees in balancing their personal and professional responsibilities.
Weekly working hours and staff rosters can be customized. We offer up to 14 weeks’ maternity leave and five days’ paternity
leave to allow employees to spend more time with their newborns. Breastfeeding rooms are also provided.
Health and well-being
Since 2019, we have been a signatory to the Joyful@Healthy Workplace Charter, launched by the Department of Health and
the Occupational Safety and Health Council to cultivate a healthy working environment. Last year, we continued to organize
talks and workshops on healthy eating and physical and mental well-being. Due to COVID-19, some of the health talks
were switched to online mode. The topics were based on staff opinion, including medicinal food therapy and personal stress
management this year. In addition, online information for home fitness activities such as yoga and stretching was provided.
Competitions were also organized to encourage employees to build a healthy lifestyle.
OUR PEOPLE
PCCW Environmental, Social and Governance Report 2021
20
PCCW maintains a multi-story sports complex housing comprehensive facilities for employees and their family. Our Sports
and Interest Group organizes various programs to continue promoting staff wellness and engagement during the pandemic.
We provide healthcare benefits and services to safeguard our staff’s health. Medical check-ups are provided for staff aged 40
or above. Our healthcare program also covers hospital and surgical benefits, an outpatient doctor plan and a supplementary
major medical plan. In 2021, we expanded our employee insurance to cover critical illness.
To encourage COVID-19 vaccination among our staff and provide adequate time for rest, we offer two days of paid leave to
employees for each dose of vaccination received. We also continue to offer a flu vaccination program at no cost to staff to
strengthen their immune response to influenza. The vaccine is offered at a discounted rate for employees’ family members.
To assist staff with personal, family or work concerns, our Employee Assistance Program provides emotional support and a
24-hour, seven-day professional counseling hotline.
Connecting with our staff
The Group publishes newsletters on the intranet on a regular basis to keep all staff up to date with the latest news and
Group-wide business developments. Through face-to-face meetings, Let’s Chat sessions and town hall style gatherings, our
employees can share feedback and suggestions with senior management. The Joint Staff Council also provides staff and
management with a forum to meet regularly and exchange ideas on operational efficiency, career development and training,
working conditions, social activities and recreational facilities.
Workplace transformation
An online workplace transformation initiative was
launched this year across the Group to enhance
employee engagement. The Workplace Transformation
project aims to facilitate users’ transition from
traditional processes into contemporary practices
that improve productivity and connection. Microsoft
365 has been chosen as a market-leading cloud
collaboration suite which supports modern ways of
working.
OUR PEOPLE
PCCW Environmental, Social and Governance Report 2021
21
2.2 Occupational Safety and Health
We maintain high occupational safety and health standards across the Group through our Statement of Safety and Health
Policy, which has been set out in accordance with the guidelines of Safety Management System. The Group’s Occupational
Safety and Health (“OSH”) committee is responsible for monitoring the relevant policy and reviewing it from time to time.
Occupational Safety and Health Council (“OSHC”) is invited to conduct a safety audit every six months, benchmarking with
Level 3 of Continual Improvement Safety Programme Recognition of System (“CISPROS”). Our safety management system
was considered as effective and efficient in the audit report.
We provide regular safety training to new and existing staff to strengthen their awareness on safe and healthy workplace
behavior. For example, they are required to report promptly to their immediate supervisors in case of any injuries or unsafe
conditions.
Safety training courses include:
• Accident investigation skills
• Certificate of Competence in Display Screen Equipment Assessment
• Confined space training
• Confined Space Certified Worker Training
• Standard First Aid Certificate Training
• Standard First Aid Certificate Refresher Training
• Work-at-height training
• Basic infection control at work
• Hazard Identification Activity Training
• Heat Stress Assessor Training
In 2021, we offered more than 300 sessions of health and safety-related training to our staff.
During the year, we encouraged our staff to learn from a simulation at the OSH Immersive Experience Hall at OSHC’s center
Tsing Yi, where they were immersed into the Cave Automatic Virtual Environment. In a safe setting featuring a dynamic
platform 4 m2
 in size, participants got to experience the scene of an industrial accident through 4D virtual reality so as to
familiarize themselves with the relevant safety measures.
OUR PEOPLE
PCCW Environmental, Social and Governance Report 2021
22
To minimize the need for staff to work at height, we introduced drones for radio cellsite inspection. Utilizing 5G, the drones
can be controlled by staff from the office with live visual monitoring of hard-to-reach locations, thereby reducing the number
of on-site personnel and enhancing staff safety.
We have appointed staff members as Designated Office Coordinators, Designated Fire Officers and First Aiders. These
individuals conduct safety inspections to eliminate hazards and provide first-aid assistance in the event of an accident. Internal
safety audits are arranged to evaluate individual BU safety management systems and physical site condition regularly.
Eligible employees are covered by our insurance against accidental death and/or permanent disablement in both workand non-work-related situations. Our work injury care program supports injured employees during their recovery and
rehabilitation, including doctor consultations and treatments.
There were zero work-related fatalities among our staff in 2021. The number of lost days due to work injury in the past three
years are as follows:
Year Work-related fatalities No. of lost days
2021 0
2020 0
2019 0
1,374 days
2,564 days
2,465 days
OUR PEOPLE
PCCW Environmental, Social and Governance Report 2021
23
2.3 Talent Retention and Development
The Group supports its workforce through comprehensive talent development programs and succession planning, enabling
our employees to grow and add value to our business.
Training and development
We conduct training and leadership programs for our staff to enhance their professional and personal development. During
the year, we transformed in-person training into virtual experiences through online platforms and webinars.
Average training hours by gender* Average training hours by employee category*
Male
13.5 hours Female
11.1 hours Below middle
16 hours management
Senior management
5.4 hours Middle management
4.6 hours
* Breakdown of average training hours is calculated by dividing the total training hours for each category by the number of employees at year end (excluding
part-time and temporary staff).
Percentage of employees trained
by gender+
Percentage of employees trained
by employee category+
Male
74% Female
64% Below middle
management 79%
Senior management
47% Middle management
64%
+ Breakdown of employees trained in relevant categories is calculated as a percentage of the total number of employees in that category at year end (excluding
part-time and temporary staff).
OUR PEOPLE
PCCW Environmental, Social and Governance Report 2021
24
New staff are introduced to our basic operations via e-orientation activities. We also organize online training on topics such as
fraud and cybersecurity for all of our full-time employees.
Two in-house monthly training programs, the Supervisory Development Program and the Managerial Development Program,
continue to strengthen the leadership and people management skills among our supervisors and leaders.
Our Future Leaders Development Program helps middle-management staff enhance their innovative and entrepreneurial
thinking.
PCCW also offers a Graduate Trainee Program to groom high-caliber graduates to become future leaders in the technology
sector. We recruit fresh graduates from engineering, IT, customer service, sales, marketing, and media disciplines.
5G Growth Model – Realigned Performance Management Model
To cultivate a performance-driven culture, a new 5G Growth Performance Management Model was launched
this year to support managers and employees in goal-setting and performance evaluation against our critical key
business goals. The five key business goals, adapted from previous years, include financials, customer experience,
operations sustainability and innovation, with the addition of a new goal on people and organization to foster better
collaboration, culture cultivation, employee development and engagement.
A series of training for all people managers on how to understand and utilize the model to coach their team members
was virtually delivered. Subsequent training sessions were conducted to support managers and employees on various
performance management activities.
To support students in their career planning, PCCW has collaborated with Vocational Training Council Group in offering
internship opportunities at our Engineering Department through the Earn & Learn Scheme since 2015. PCCW also provides
attachment opportunities for Hong Kong Institute of Vocational Education (“IVE”) Higher Diploma students in Engineering
and IT disciplines.
To support business growth and to allow greater agility, we proactively review and implement training and development
initiatives that are both timely and fit for purpose. In addition to conventional training, other learning and development
methods are adopted such as virtual learning, peer-to-peer learning, coaching and mentoring.
Talent attraction and retention
A Group-level performance appraisal system and incentive bonus schemes are in place to motivate and reward employees. To
further enhance the capabilities of our staff and facilitate developmental discussion between employees and managers, we
have revamped our human resources (“HR”) system and learning platform.
Turnover rate*
32.46% 2019
2021
26.12% 2020
36.8%
* Turnover rate covers voluntary leavers only.
OUR PEOPLE
PCCW Environmental, Social and Governance Report 2021
25
Employee turnover rate by gender, age group and geographical region in 2021*
By gender By age group
Male
36% Female
38% Above 50
<30
30–50
8%
25%
83%
By geographical region
Hong Kong 24%
China excluding Hong Kong 73%
Others 32%
* Breakdown of turnover rate is calculated by dividing the number of voluntary leavers for each category by the yearly average number of employees in that
category.
New HR System – “Connect”
This year, PCCW launched “Connect”, a new HR information system supported by SAP SuccessFactor, as the singular
source for key HR data and processes. The system allows employees to access HR information more readily through an
enhanced, digitalized experience, and for supervisors to manage their teams more effectively.
The system is designed with functionalities to support employees on their daily HR activities, such as searching for
contact details of colleagues through our employee directories, and applying for and approving leave. Apart from
basic employment functions, the system also provides a simplified and digitalized platform for managing performance
including goal-setting and evaluation, as well as identifying and applying for internal opportunities and job opening
referrals.
By adopting the system, HR procedures are digitalized to provide real-time talent data for workforce planning and
informed decision making which improve our employees’ HR lifecycle experience. We believe that through this HR
system, our business effectiveness and efficiency can be improved by lowering the overall cost of administration
work and mitigating the compliance gaps and challenges related to HR processes and talent data. We also hope to
leverage this platform to facilitate our internal communication and feedback among staff in order to further improve
performance.
OUR PEOPLE
PCCW Environmental, Social and Governance Report 2021
26
2.4 Diversity and Inclusiveness
We embrace diversity and inclusion in the workplace. As part of our commitment, we have assumed the role of signatory to
the Racial Diversity and Inclusion Charter for Employers under the Equal Opportunities Commission.
PCCW’s diverse talent pool comprises employees of over 50 nationalities with various expertise and background. We are
dedicated to providing equal opportunities for all employees in various employment aspects, including remuneration,
recruitment, training and promotion. We prohibit all forms of discrimination based on gender, age, family status, sexual
orientation, disability, race and religion. Among our leadership positions, over 30% of roles are currently filled by female
staff. As at the end of 2021, there were 36 disabled persons working at PCCW.
20,693 employees
from
over50
nationalities
We uphold our labor standards as stipulated in our CSR and HR policies. Child and forced labor is strictly prohibited in our
business operations.
Senior management
Male
2.67%
Senior management
Female
1.31%
Middle
management
Male
20.40%
Middle
management
Female
9.37%
Below middle
management
Male
35.44%
Below middle
management
Female
30.81%
Total employees by employment category
Part-time employees
(including temporary)
Male
2.94%
Part-time employees
(including temporary)
Female
5.40%
Full-time
employees
(including
contract)
Male
55.57%
Full-time
employees
(including
contract)
Female
36.09%
OUR PEOPLE
PCCW Environmental, Social and Governance Report 2021
27
Total workforce by age group
<30
28.02%
>50
15.3%
30–50
56.68%
Total workforce by gender
Male
58.50%
Female
41.50%
Total workforce by
geographical location
Others
10.76%
Hong Kong
63.34%
United States
0.68%
China excluding
Hong Kong
25.22%
3. OUR ENVIRONMENT
OUR ENVIRONMENT
PCCW Environmental, Social and Governance Report 2021
28
PCCW has made continuous efforts to build a more sustainable business and help address the threat of climate change. We
have adopted a wide range of mitigation and adaptation measures on energy saving, waste management, sustainable use of
resources and smart city development to help achieve a low-carbon economy.
We conduct our business in accordance with the applicable environmental laws and regulations. These include the Energy
Efficiency (Labelling of Products) Ordinance (Cap. 598), Product Eco-responsibility Ordinance (Cap. 603), Product Ecoresponsibility (Regulated Electrical Equipment) Regulation (Cap. 603B), and Buildings Energy Efficiency Ordinance (Cap. 610).
We have also established internal standards such as the Energy and Water Management Policy and Guidelines, and recycling
procedures and programs.
Objectives
Minimize energy consumption and GHG
emissions
Promote responsible waste
management
Help employees and customers
become more environmentally
friendly
Measures
Set environmental targets and track performance
Modernize exchange buildings, equipment and
infrastructure
Upgrade and electrify our fleet
Promote recycling
Develop green ICT solutions to optimize and reduce
resource consumption
Environmental performance highlights
Electricity consumption Total GHG emissions General waste2
 disposal
2 General waste mainly includes general office waste
358,192,525 kWh
( 3.47%)
194,203 CO2e
( 18.30%)
824.47 tonnes
( 7.55%)
Water consumption
366,000 m3
( 7.10%)
OUR ENVIRONMENT
PCCW Environmental, Social and Governance Report 2021
29
The calculation of GHG emissions follows the procedures set out in the Guidance on Climate Disclosures (“Guide”) of the
HKEX using the emission factors provided by the power companies. The emission factors are factors by which electricity use
is converted to GHG emissions. CLP and HKE reduced their factors by 26% and 12.4% respectively which led to a notable
decrease in our GHG emissions in 2021.
On the other hand, general waste disposal and water consumption increased in 2021 as a result of increased number of days
working in the office given the alleviated situation of COVID-19, as compared to 2020 during which the Group’s staff spent
more time working from home.
3.1 Climate Change, Energy Consumption and GHG Emissions
Climate change can significantly affect our business operations if the relevant risks are not assessed properly. An increase in
temperature may lead to higher electricity consumption for cooling; extreme weather events, such as super typhoons, may
cause physical damage to our submarine cables and other infrastructure and result in financial loss.
PCCW understands the importance of enhancing energy efficiency and reducing our carbon footprint in our daily business
operation to combat climate change. Our Environmental Advisory Group meets regularly to evaluate our sustainability
agenda. The Risk Management, Controls and Compliance Committee continuously assesses the impact of climate change,
which is currently considered an emerging risk to the Group. During the year, we have been studying the feasibility of
conducting the first phase of a climate risk assessment in a bid to build a more sustainable business and address the
impending threats.
We have been voluntarily disclosing our carbon emissions data to the Carbon Footprint Repository for Listed Companies in
Hong Kong since 2014. Launched by the Environmental Protection Department, the repository encourages listed companies
to disclose their GHG emissions and the carbon reduction measures implemented.
To mitigate the effects of global warming, we have been using eco-friendly refrigerants in our new air-conditioning systems.
We have also followed the guidance of the Montreal Protocol to phase out ozone-depleting hydrochlorofluorocarbons
(“HCFCs”).
HKT sustainability-linked loans
As a way of embedding sustainability values into our business strategies, HKT has raised about
US$1 billion in sustainability-linked loan facilities since 2020. The interest margin of the loans is
linked to designated sustainability performance targets. Apart from supporting the development
of sustainable financing, we are determined to drive long-term sustainability enhancements and
reduce the climate impact of our operations.
Improving energy efficiency
The most energy-consuming elements of our facilities are our infrastructure, exchange buildings, telecom and IT equipment,
and offices. We have formulated a policy for maintaining the temperature of offices, buildings and general facilities between
24°C and 26°C. We also review our exchange buildings’ management systems and energy consumption quarterly, and
minimize electricity consumption by upgrading equipment and facilities.
OUR ENVIRONMENT
PCCW Environmental, Social and Governance Report 2021
30
We have adopted the following measures:
• Phase out old legacy equipment by using new systems with improved energy efficiency
• Replace fluorescent tubes with LED lights
• Install LED lights at new premises
• Install occupancy sensors for lighting control in staircases and carparks
• Replace air-cooled chillers with water-cooled models
• Review and adjust the operating control of chiller systems
• Modernization of lifts
• Switch off non-essential display monitors in the 24-hour operation centers
• Shorten the operating hours of air conditioners in some offices
• Consolidate duty staff to centralized working areas on Saturdays and public holidays
PCCW Solutions data centers are designed and maintained to the highest environmental standards. All power supplies
including Uninterruptible Power Supply, air-conditioning systems, backup generators and other electrical and mechanical
signaling services for our facilities have adopted the most advanced environmentally friendly technologies and measures. Our
data center efforts are recognized by Leadership in Energy and Environmental Design (“LEED”) Platinum accreditation and
ISO 14001 certification for Environmental Management System. We have continuously improved our Data Center Power Usage
Effectiveness to attain best-in-class performance levels in the region.
We have been a signatory to the Charter on External Lighting since 2016. In addition, we also continued to take part in the
Energy Saving Charter and have pledged to adopt energy-saving practices in our exchange buildings and the shops of csl,
1O1O and HKT.
With our concerted efforts, the Group’s electricity consumption was reduced by more than 12.9 GWh in 2021.
First solar power systems commissioned
In addition to conserving energy, we have joined CLP’s Renewable Energy Feed-in Tariff (“FiT”) Scheme to help promote
renewable energy use. The Group’s first solar power system, with a capacity of 10 kW and located on the rooftop of our
Junk Bay Exchange, was commissioned in May 2021.
An even larger solar power system in Tin Shui Wai was put into operation in November 2021. An estimated
120 MWh of electricity can be generated annually from both locations, sufficient to power 1,520 9W LED light bulbs for
a year.
Plans to expand the system to exchanges in other areas of the city are currently under review.
OUR ENVIRONMENT
PCCW Environmental, Social and Governance Report 2021
31
In 2021, we continued with upgrades to our vehicle fleet. A total of seven Euro 6 vehicles and eight electric cars were
introduced for a potential reduction of 34 tonnes of CO2 emissions per year. We plan on upgrading 10 more vehicles, which
account for some 4.3% of our fleet, in the upcoming three years.
3.2 Sustainable Use of Resources
We strive to be a good steward for natural resources and adopt green operation practices whenever possible. Electronics,
packaging and general waste are the main sources of waste generated throughout our operations and value chain.
Paper use
Throughout the years, we have continued to promote digitalization among our customers. In 2021, we saved over 47 million
sheets of paper by encouraging our customers3
 to switch to electronic bills for our various services. The percentages of
customers using e-billing are as follows:
NETVIGATOR Mobile Now TV Fixed-line, eye, IDD
98% 97% 85% 64%4
As photocopying is a major source of paper consumption in the Group, we use paper certified under the Programme for the
Endorsement of Forest Certification for photocopying and bill printing.
3 This refers to consumer customers.
4 e-Billing service was first offered to fixed-line, eye and IDD customers in November 2016. The percentage of customers opting for e-billing increased from
about 30% in 2017 to 64% in 2021.
OUR ENVIRONMENT
PCCW Environmental, Social and Governance Report 2021
32
Waste management
We follow the principle of waste hierarchy, including reuse, recycling, reprocessing and responsible waste disposal, in order
to better manage the waste generated. We regularly evaluate the effectiveness of our waste management approach to
determine the best options that create minimal impact on the environment.
Strict waste management instructions have been put in place to ensure proper waste
disposal. Hazardous waste such as fluorescent tubes, industrial batteries, waste electrical
and electronic equipment (“WEEE”) and general office batteries are handled by approved
chemical waste collectors and specialist contractors, while non-hazardous waste is
handled by professional cleaning service providers and contractors in compliance with
local regulations. The two main types of non-hazardous waste include general office
waste and construction waste from the renovation of our retail outlets. Our office
furniture is reused following shop relocation and renovation whenever possible to
minimize waste generation.
Since 2019, we have collaborated with Hong Kong Battery Recycling Centre to
recycle waste lead acid batteries locally.
To support the Government’s initiative to reduce disposable plastic tableware
consumption, our staff canteens no longer offered disposable plastic cutlery on
Wednesdays. Colleagues are encouraged to bring their own cutlery to the office.
Our staff canteens no longer provide plastic straws and cutlery starting February 2022.
e-Waste management
To align with the Government’s Producer Responsibility Scheme for WEEE, PCCW provides removal services for our customers
whenever they purchase regulated electrical equipment. The collected waste equipment is sent to certified recyclers for
proper treatment to achieve resource recovery. In 2021, we helped customers remove more than 2,500 pieces of WEEE.
By introducing in our retail shops a trade-in and preliminary valuation service, we also encourage customers to reduce waste
generation and promote a circular economy when they replace their mobile devices. In 2021, we collected and recycled over
5,500 old mobile handsets and accessories from customers for donation to Caritas Computer Workshops.
“Say Goodbye to Disposable Plastic”
Poster
OUR ENVIRONMENT
PCCW Environmental, Social and Governance Report 2021
33
Recycling and reuse
We organize recycling programs for employees and customers to facilitate the recycling and reuse of waste materials. These
include toner and ink cartridges, scrap materials, copper, iron, steel, and paper. We complement our efforts by donating
obsolete IT products such as computers and printers to charitable organizations.
Recycled items 2019 2020 2021 Change
(2021 vs 2020)
Toner and ink cartridges
(pieces)
1,808 1,454 1,152 -20.77%
Scrap materials (pieces) 230,145 191,026 243,713 +27.58%
Copper (tonnes) 18.67 13.53 10.19 -24.69%
Iron and steel (tonnes) 7.71 12.70 1.44 -88.66%
Paper (tonnes)5 142.20 107.06 118.48 +10.67%
Guided by our Energy and Water Management Policy and Guidelines, PCCW is committed to conserving, reducing and
reusing water in our operations. We noticed that our exchange buildings consume the most water in their operation.
Therefore, the wastewater from water-cooled condensers is collected and used for flushing, which helps reduce our fresh
water consumption. In addition, automatic faucets and toilet flushers are installed in our buildings at the earliest opportunity
possible.
Cross-industry recognition
Our efforts in promoting sustainability have been recognized by external parties, which serves as motivation for us to take
greater initiative in doing our part for the environment.
Accolade Awarding organization
Friends of EcoPark Award EcoPark Management Company & Environmental Protection
Department
Silver Award (Media and Communication
Sector) 2020
The Hong Kong Awards for Environmental Excellence, led by
Environmental Campaign Committee & Environmental Protection
Department
Hong Kong Sustainability Award 2020/21:
Certificate of Excellence
Hong Kong Management Association
5 Paper recycling data in 2019 and 2020 have been restated after data review.
OUR ENVIRONMENT
PCCW Environmental, Social and Governance Report 2021
34
3.3 Environmental Targets
In order to fulfill the latest requirements listed in HKEX ESG Reporting Guide, address investors’ increasing expectations on
public disclosure of environmental performance and targets, as well as enhance the Group’s ESG performance, we have
established environmental targets during the reporting period. The current quantitative targets cover the period up to 2025.
Referencing the results of historical data analysis, internal operational review, peer benchmarking and external context
review, the following targets have been set:
Area of
reduction
Baseline Base
year
Target
year
Target
type Target
Electricity consumption 390,591,712 kWh 2018 2025 Absolute -13.2%
GHG emissions
(Scope 1 & 2)
248,912 tonnes CO2e 2018 2025 Absolute -34.3%
Water consumption 22.10 m3
/employee 20196 2025 Intensity -4.4%
General waste7 860.51 tonnes 2018 2025 Absolute -16.2%
We will continue to monitor and track our performance in the areas of resource consumption as well as waste and emission
reduction against these targets. To strive for improvement in environmental performance, longer-term reduction targets will
be formulated and disclosed at an appropriate time going forward. It is also our intention to align with the Government’s
proposal to establish general waste diversion rate targets in the future.
6 Because of a major water waste leakage incident in 2018, the water consumption was abnormal for that year. Therefore, 2019 is chosen as the base year
for the water consumption reduction targets.
7 General waste mainly includes general office waste.
OUR ENVIRONMENT
PCCW Environmental, Social and Governance Report 2021
35
3.4 Building a Smart City
We embrace digital transformation by infusing green components into our products and services, assisting our clients in
adopting a sustainable lifestyle and business solutions, and contributing to the development of a smart city.
Smart Charge – one-stop EV charging solution
We have been partnering with CLP since 2016 to present electric vehicle (“EV”) owners with hassle-free EV-charging
solutions. In addition to sourcing for the most suitable equipment and fielding highly experienced technicians, Smart
Charge (HK) Limited negotiates with building management to optimize installation of EV-charging facilities. All of our
services are backed up by 24/7 support to instill the utmost confidence and trust in our customers.

Energy Optimization with IoT
PCCW also leverages emerging technologies to empower
enterprises to achieve energy-saving objectives.
Energy Optimization with IoT is one such initiative.
By optimizing electric current, heat and power loss is
reduced, thereby attaining energy conservation. This
solution has already been deployed in a commercial
building in East Kowloon. A double-digit percentage
in power savings has been achieved in existing
infrastructures.
OUR ENVIRONMENT
PCCW Environmental, Social and Governance Report 2021
36
Energy control through artificial intelligence and machine learning
Energy control is one of the applications for Artificial
Intelligence (“AI”) and Machine Learning (“ML”). By utilizing
such technology, the Company has enabled management of
office lighting, air-conditioning and smart services through
apps and sensors with energy-monitoring functions. This has
been adopted at our North Lantau site office and applied to
300 lighting luminaries and 100 air-conditioning units. The
algorithms significantly improved energy efficiency, successfully
reducing aggregate energy consumption by nearly 20%.
3.5 Employee Environmental Awareness
We actively promote environmental awareness among our employees through various staff communication channels and
encourage their participation in green activities.
PCCW has been a signatory to the Green Mid-Autumn Festival Food Saving Pledge initiated by Food Grace since 2018.
During the year, we collected surplus mooncakes for those in need, especially low-income families and elderly persons living
alone.
To promote food waste reduction and exercise our corporate social responsibility, we participated in Bread Rescue, which
was launched by a bakery chain in Hong Kong. In support of the program, our staff volunteers collected surplus bread for
redistribution to NGOs and the underprivileged.
We also continued our support to WWF Earth Hour by switching off signage lighting in 13 office buildings, exchanges and
shops for one hour on March 27.
Our volunteers participated in The Green Earth’s Plantation Enrichment Program,
assisting with tree care and maintenance activities such as weeding and
fertilizing at Tai Lam Country Park. The program aims to enhance the biodiversity
and ecological value of plants in country parks by planting native tree seedlings
in existing woodlands. We also joined their Waste Hunting in the Wild event to
collect rubbish along a hiking trail near Aberdeen Reservoir.
It is our belief that effective communication is vital to fostering an understanding
of sustainability values. Thus, we publish a column called Green Matters in
our internal newsletter on a regular basis to provide green tips and keep our
colleagues informed about new sustainability projects and trends. A dedicated
email account is used to collect feedback and suggestions for improvement from
colleagues.
Green Tips for post-Chinese New Year in
Green Matters
4. OUR COMMUNITY
OUR COMMUNITY
PCCW Environmental, Social and Governance Report 2021
37
PCCW is committed to creating positive social impact through a variety of community service initiatives. We continually
identify and support social causes through financial donations and in-kind contributions, education and corporate
volunteering services.
Objectives
Support vulnerable and underprivileged groups
Digital empowerment
Leverage technology to improve
quality of life
Respond to community needs
amid the new normal
Measures
Corporate volunteering in community service
projects
Telecom services sponsorships
Programs and workshops for youth and the elderly
Initiatives to support smart city evolution
Guided by the Group’s CSR Policy, we focus on addressing the needs of local communities. We partner
with charitable organizations, leveraging our resources and ICT expertise to support underprivileged
groups and build a more inclusive society. We also utilize technologies in promoting active and smart
aging and enabling students and youth to engage in the digital world.
Our work in the community
The COVID-19 situation in 2021 continued to force the cancellation and postponement of many
community service programs and activities.
5,715
196
200
10
231 3,914
11 HK$20M+
Cumulative number of
registered volunteers since 1995:
Number of partnering
NGOs and academic
institutions in 2021:
Volunteer leave days
granted in 2021:
Value of monetary donations
and in-kind sponsorships for
charitable causes in 2021:
Special community service
programs in 2021:
Ongoing community service
programs in 2021:
Active volunteers
in 2021:
Volunteer hours
in 2021:
OUR COMMUNITY
PCCW Environmental, Social and Governance Report 2021
38
4.1 Community Engagement
Our Corporate Volunteer Team is composed of our employees, their family members, and company retirees, who have been
working together to build a better community for the past 26 years. In 2021, our Team contributed close to 4,000 hours of
service.
PCCW organizes the annual Volunteer Award Ceremony to recognize our staff volunteers’ valuable contributions. The 2021
ceremony was held virtually in July, with the presence of guests from the Social Welfare Department and our NGO partners.
Through implementing the Volunteer Appreciation Scheme, we encourage employees to participate in community service by
awarding up to two days of volunteer leave each year.
In 2021:
• PCCW continued to be awarded the 15 Years Plus Caring Company Logo under Hong Kong Council of Social Service’s
Caring Company Scheme, in recognition for being a caring company for 19 years.
• PCCW was recognized in the Social Capital Builder Logo Awards under the Labour and Welfare Bureau’s Community
Investment and Inclusion Fund
• Now TV received Best Corporate Social Responsibility Media – Bronze in Sparks Awards 2021 organized by Marketing
Interactive
OUR COMMUNITY
PCCW Environmental, Social and Governance Report 2021
39
Community service highlights in 2021
PCCW provides a large variety of community services to different beneficiaries, including the elderly, students and youth,
children, the jobless and homeless, and people with disabilities, among others, through our community partners, including
the Government, NGOs, academic institutions, and other organizations. In 2021, the Corporate Volunteer Team held 21
programs in partnership with charitable organizations and social service groups.
Smartphones for Needy Elders
csl has provided 100
phones and two-year
mobile plan sponsorships
to the elderly serviced
by the Neighbourhood
Advice-Action Council and
Mighty Oaks Foundation.
Our volunteers trained
a group of secondary
school students to provide
smartphone operation tips
for the elderly.
OUR COMMUNITY
PCCW Environmental, Social and Governance Report 2021
40
The elderly
• HKT’s elderly hotline continued to provide timely technical support to citizens aged 65 or above, helping with contracts
and bills as well as service relocation. Special concessions for home phone, broadband and mobile services are available
for eligible applicants from low-income families.
• The Group has been supporting the Dragon Boat Festival Elderly Care Program for 19 years. In June 2021, 115 volunteers
delivered rice dumplings and anti-pandemic supplies to elderly residents of Wong Tai Sin.
• DrGo and Quality HealthCare jointly provided chronic disease caregivers from South Kwai Chung Social Service with 100
free remote counseling sessions on the DrGo platform.
• In September, close to 110 volunteers and their family shared in the joy of celebrating Mid-Autumn Festival with 360
elderly residents of Choi Hung Estate during a visit.
• Now E partnered with HKJC Centre for Positive Ageing to offer Oscar-winning movie The Father to Hong Kong viewers in
a bid to raise public awareness of Alzheimer’s disease during World Alzheimer’s Month.
Supporting the elderly community amidst the COVID-19 pandemic
DrGo and csl have collaborated with Hong Kong Jockey Club Charities Trust, South Kwai Chung Social Service, and
Precious Blood Hospital (Caritas) to organize the “Elderly Care Anti-pandemic Program” (愛在樂齡抗疫計劃) since July
2021.
The scheme provides remote healthcare and anti-pandemic services for elderly patients, benefiting over 450 seniors
in Kwai Tsing district. Our Corporate Volunteer Team fully supports the scheme by assisting in the operation of virtual
health consultations, the collection of health surveys, and blood pressure checks for the elderly.
OUR COMMUNITY
PCCW Environmental, Social and Governance Report 2021
41
Launching one-stop service to assist elderly in using LeaveHomeSafe
In response to the Government’s anti-pandemic measures, csl launched a one-stop service to help the elderly use the
LeaveHomeSafe mobile app. We provided assistance by organizing smartphone workshops and setting up a 24-hour
service hotline, as well as providing affordable smartphones and mobile service plans.
We have dedicated ambassadors at HKT and csl stores to help the elderly install the LeaveHomeSafe app and show
them how to scan QR codes. In addition, our ambassadors also teach smartphone operations such as setting up
personalized interfaces and installing new apps for everyday communication and entertainment.
OUR COMMUNITY
PCCW Environmental, Social and Governance Report 2021
42
Children and youth
• The Group has supported the Child Development Fund mentorship program organized by Lok Sin Tong from 2020
to 2022. Staff volunteers become life coaches for upper primary students from two schools to widen their horizons,
providing mentorship and companionship throughout the three-year program.
• Our volunteers organized the STEM Experience Day in collaboration with Yaumati Kaifong Association School for ethnic
minority students to learn basic coding and programming.
• HKT provided complimentary one-year broadband service to designated students referred by Methodist Centre.
Support for underprivileged students during social distancing
We also supported Caritas Hong Kong’s Caritas Grassroots Connected
Programme by providing one-year broadband service to 800 low-income
families to help address the online learning difficulties faced by students living in
subdivided flats.
OUR COMMUNITY
PCCW Environmental, Social and Governance Report 2021
43
People with disabilities
• We partnered with Salvation Army PATH Centre to launch Teen Hey Buddies Mentorship Programme, offering different
workshops for people with autism.
• In support of the Jockey Club Sports Programmes with Audio-description Service, Now TV provided the broadcasting feed
of EURO 2020 to the Hong Kong Blind Union, enabling them to provide audio description service for their members to
enjoy the selected EURO 2020 matches.
Environmental protection
• Now TV selected ten programs in celebration of Earth Day to promote environmental protection.
• Our volunteers participated in the Plantation Enrichment Programme organized by The Green Earth to help perform tree
care and maintenance work such as weeding and fertilizing at Tai Lam Country Park. The program aims to increase the
biodiversity and ecological value of plants in country parks.
• Our volunteers also participated in Waste Hunting in the Wild organized by The Green Earth to collect trash along a
hiking trail near Aberdeen Reservoir, with the aim of rediscovering the beauty of nature.
OUR COMMUNITY
PCCW Environmental, Social and Governance Report 2021
44
Others
• PCCW participated in the Bread Rescue initiative in early December to provide food for underprivileged groups.
• PCCW participated in the Hong Kong Cancer Fund’s Dress Pink Day to increase awareness of breast cancer and raise
funds for cancer care services.
• DrGo organized the Never Give Up Online Charity Concert, the proceeds of which were donated to The Mental Health
Association of Hong Kong.
• PCCW became a recognized company in the Inaugural SportsHour Company Scheme to encourage employees and their
family members’ participation in one hour of physical activity daily to foster a healthy lifestyle.
• To enable digital inclusiveness within our community, HKT partnered with the Hong Kong Council of Social Service’s
WebOrganic to provide broadband service concessions for persons with disabilities, students and the elderly from lowincome families.
Partnership with Quality HealthCare to launch mental health video consultations
Given the increasing public awareness of mental health
issues, HKT and Quality HealthCare Medical Services
launched mental health consultations including counseling,
psychotherapy and psychiatric services on the DrGo
platform, providing DrGo users with access to specialty
services, thereby empowering them to manage both their
physical and mental well-being.
OUR COMMUNITY
PCCW Environmental, Social and Governance Report 2021
45
Philanthropic sponsorship
We support charities and other organizations sharing the same goals as us through sponsorships and donations. HKT’s
online CSR platform, Club Hope, helps increase public awareness of communities in need and raise funds for them. Currently
the platform supports 13 charity organizations in six categories, namely animal welfare, disability and special needs, eco
and social caring, elderly care, music and arts. Every contribution that donors make goes to the charities of their choice. In
May 2021, as a token of our appreciation, the top 10 donors in terms of donation amount were invited to the rehearsal of
MIRROR “ONE & ALL” Live 2021 concert.
Last year we helped the community face the challenges brought by COVID-19 via various initiatives, including the provision of
free mobile data and broadband service. In 2021, PCCW contributed over HK$20 million in monetary donations and in-kind
sponsorship. Regular sponsorship for hardware and communications services included:
• Silver Sponsorship for Walk For Equality Charity Fundraiser, organized by SENsational Foundation
• Telephone hotline support for the fundraising TV shows of the Tung Wah Group of Hospitals, Po Leung Kuk, Yan Chai
Hospital and Yan Oi Tong
• Scholarships and bursaries to six local universities to support students of computer science, IT and related disciplines for
the academic year. To drive more female participation in the technology sector, we have designated the inclusion of
at least one sponsorship for female students per university this year. Extensions to include other universities are under
consideration for 2022
• Consultation service hotlines for The Samaritan Befrienders Hong Kong, Hok Yau Club, Hong Kong Children & Youth
Services, Hong Kong Sheng Kung Hui and Tai Hang Youth Centre
• St. James Settlement’s Grant-in-aid Brightens Children’s Lives Service to equip disadvantaged children with diverse
learning resources, sponsored by Now TV
OUR COMMUNITY
PCCW Environmental, Social and Governance Report 2021
46
4.2 Digital Empowerment
PCCW strives to apply its expertise and resources in digital technologies to conduct various research and development (“R&D”)
projects related to smart city initiatives, cloud applications, Big Data Analytics and AI, cybersecurity, and mobile network
innovations.
• The HKT Innovation Lab launched the HKT Startup Ecosystem in November 2020, providing a platform for driving
innovation and R&D on products and services. In 2021, through connecting with organizations from various fields, the
platform continued to provide opportunities and resources to nurture young technopreneurs and generate innovative
ideas for more effective productization, service implementation and business operation.
• Now TV sponsored 200 free two-month STEM Learning Pack passes for members of The Boys’ and Girls’ Clubs Association
of Hong Kong participating in the first STEM Awards Scheme.
• HKT partnered with Lingnan University to organize a visit to HKT Smart City Operation Center, allowing students to gain
first-hand knowledge of emerging technologies such as 5G, cloud computing, and other smart applications.
• HKT sponsored the PolyU Innovation Challenge, a start-up ideas competition for The Hong Kong Polytechnic University
students to come up with new technological ideas and business models under the main theme of “Smart City”.
5. OUR CUSTOMERS
OUR CUSTOMERS
PCCW Environmental, Social and Governance Report 2021
47
To build a long-term relationship with our customers, PCCW endeavors to provide exceptional
customer experience and high-quality products and services, including fixed-line, broadband, mobile
communication, media entertainment and other innovative offerings. We also strive to help our
customers make informed decisions by providing accurate and transparent information on our products
and services.
Objectives
Safeguard personal data
Provide reliable quality services
and products
Meet and anticipate
customer needs
Maintain high-quality
customer service
Measures
Implement privacy and personal data policies
Meet and exceed performance targets
Continuous innovation
Promote customer service excellence
We have in place stringent internal policies on customer privacy, labeling and advertising. We regularly monitor relevant new
laws and regulations so that we can communicate them in a timely manner to the responsible operational units.
During the reporting period, there was no breach of relevant laws and regulations, including but not limited to the Personal
Data (Privacy) Ordinance (Cap. 486), EU’s General Data Protection Regulation (“GDPR”), the Telecommunications Ordinance
(Cap. 106), the Broadcasting Ordinance (Cap. 562) and the licence conditions and code of practice issued by the Office of the
Communications Authority (“OFCA”).
5.1 Customer Data Privacy and Security
Customer data privacy and protection is ranked as the most prominent topic in our materiality review. To address stakeholder
concern and to fulfill legal requirements, we uphold the highest standards in protecting customer data privacy. We strictly
follow our internal policies, procedures and compliance guidelines governing how we collect, use and manage customers’
information. These clearly define the roles and responsibilities of our staff in handling personal data, and stipulate appropriate
security measures to achieve confidentiality, integrity and accountability. The policies and guidelines are reviewed periodically
to ensure PCCW is up to date with the latest regulations, technology, and industry best practices. In 2021, there were no
known issues of non-compliance in this area.
At the Group level, the Group Information and Cybersecurity Council (“GICSC”) oversees all cybersecurity-related initiatives,
investments and ongoing maintenance pertaining to the protection of the Group’s core network, servers and endpoints. The
GICSC reports directly to top management on any matter requiring escalation. Moreover, there are dedicated teams under
Group Risk Management and Compliance overseeing technology risk management and data privacy compliance across the
Group. The teams are responsible for maintaining robust controls and proactive enhancements as well as investment in
security management to enable effective response on cybersecurity issues, if any.
OUR CUSTOMERS
PCCW Environmental, Social and Governance Report 2021
48
At department level, some of our BUs and functions have obtained the ISO 27001 (Information security management)
accreditation, demonstrating our effort in data protection and management. PCCW Solutions has obtained ISO 27701 Privacy
Information Management System certification, which is an extension of ISO 27001 (Information security management).
This demonstrates our full commitment to data and information security and complying to privacy regulations, in particular
Personal Identifiable Information (“PII”). In addition, all new employees are required to complete mandatory training on data
privacy as part of their induction. Employees with access to personal data are also provided with annual refresher privacy
training. External cybersecurity awareness training and exercise (e.g. phishing test) are also held.
During the year, the Group set up a Data Breach Response Plan to enhance data breach handling while enabling prompt
notification to stakeholders.
Information security and management
To identify and manage emerging information security risks, PCCW’s management assesses business
strategy, new technologies, customer concerns and relevant industry developments on a regular
basis. The Group Information and Cybersecurity Office (“GICSO”) is responsible for reviewing
the overall cybersecurity risk profile and monitoring suspicious traffic and activity to combat
cyberattacks. We constantly review the latest development on cybersecurity to enhance our policies
and investment in capabilities and technologies to be well-equipped for timely response in the case of any newly identified
risk. A Data Protection Impact Analysis (“DPIA”) is conducted before we enter into business in a new country or introduce
any new product or service. The DPIA identifies data privacy risks in the business process, provides a basis on which to assess
and implement the corresponding risk mitigating controls, and ensures our compliance with all data protection obligations.
Over the years, we have progressively extended the coverage of our next-generation endpoint protection across the Group to
further enhance data security. Anti-virus software, network behavior tools, threat intelligence exchange and advanced threat
defense infrastructure are also in place to enhance our cybersecurity.
Cybersecurity measures for customers
HKT provides the NETVIGATOR SHiELD cybersecurity service for our broadband customers to protect IoT devices against
phishing, malicious sites and potential botnet connections. Cybersecurity incidents are closely monitored by our network
engineering team so that responsive actions can be taken when necessary. A two-step verification procedure has been
launched to strengthen the security of our customer email accounts. To enhance customer awareness on cybersecurity,
NETVIGATOR provides customers with regular updates on ways to identify suspicious content, calls and activities through the
Safe Internet Tips and Customer News channels and the NETVIGATOR and customer service Facebook page.
During the year, HKT sponsored Hong Kong Productivity Council in the release of the HKT Hong Kong Enterprise Cyber
Security Readiness Index 2021 to help the public understand the latest trends in cybersecurity, raise awareness and advise
preventive measures to tackle cybersecurity threats.
In 2021, HKT Enterprise Solutions received the FinTech Awards 2020 in Cybersecurity/Anti-Fraud – Outstanding Cybersecurity
Solutions (Business) from ET Net.
OUR CUSTOMERS
PCCW Environmental, Social and Governance Report 2021
49
5.2 Reliable and Responsible Services and Products
PCCW has a set of systematic and rigorous quality management procedures in place to ensure
our services and products are safe and reliable. We have a dedicated team responsible for the
development and management of customer services and products, and strictly comply with
OFCA’s requirements.
We have acquired various international quality and management system certifications, including Hong Kong Q-mark for
our field and center operations; ISO 9001:2015 (Quality management systems); ISO 20000 (IT service management); ISO
27001:2013 (Information security management); TL 9000 (Quality management system for the telecommunications industry);
ISO 27017:2015 (Code of practice for information security controls for cloud services); and ISO 27018:2019 (Code of practice
for protection of personally identifiable information (“PII”) in public clouds for cloud services). These certifications recognize
our quality and management systems to be aligned with international best practices across various operations, including fixed
and wireless network planning and operation, cloud application and development, field services and project management. In
2021, HKQAA concluded the findings of their ISO Surveillance Visit to be satisfactory. No non-conformity was found during
the audit.
We completed the migration of local line services from digital switching technology to next-generation network technology
in 2020 for more reliable services.
We constantly monitor the quality of our products and services through a range of performance indicators across different
functional units:
Performance
target
Actual performance
in 2021
csl
Network reliability8 99% 100%
Service restoration9 < 60 minutes 100%
NETVIGATOR
Network stability10 99.99% 99.993%
Service restoration11 99% 99.885%
In 2021, we offered 23,605 hours of internal and external training to 2,120 technicians in the engineering team on topics
such as latest industry trends and developments.
We encourage our employees to obtain professional certification and accreditation in their field of expertise. In 2021, our
engineers possessed a total of 2,392 professional certificates and institutional memberships.
8 Availability of the core network or core network uptime in a set observation period.
9 Mean time for recovering a fault in the core network following its discovery and identification.
10 Availability of broadband network.
11 Provide restoration of services for customers within two calendar days.
OUR CUSTOMERS
PCCW Environmental, Social and Governance Report 2021
50
Service accessibility
PCCW is committed to enhancing social inclusion, digital accessibility and service availability to different groups in Hong
Kong.
By the end of 2021, HKT’s FTTH network coverage reached 90.4% in Hong Kong. We continued to expand our service
coverage, in particular by providing reliable broadband services in remote regions. Taking into account our 5G WTTH service,
a total of 98% of homes are covered.
Supporting the Government’s initiative to extend fiber-based networks to remote villages and exploring opportunities to
extend our broadband coverage, HKT has completed network rollout works in 16 villages under OFCA’s Subsidy Scheme to
Extend Fibre-based Networks to Villages in Remote Areas in 2021. The projects, awarded to HKT in 2019/20, cover a total of
97 villages across Tai Po District, Sai Kung District, Lamma Island, Lantau Island, Cheung Chau and Peng Chau.
We possess more than 3,000 mobile sites in Hong Kong, covering all transportation tunnels and railway lines, and indoor and
outdoor areas in major universities.
In terms of Wi-Fi coverage, we offered 21,751 Wi-Fi hotspots in Hong Kong as at the end of 2021, providing comprehensive
coverage for locations such as convenience stores, restaurants, MTR stations and public phone kiosks.
FTTH coverage:
90.4%
FTTH + WTTH coverage:
98%
Number of Wi-Fi hotspots in Hong Kong:
21,751
We have devoted resources to address the special needs of vulnerable groups in society by offering helpful solutions to
people in need. Our retail shops are equipped with barrier-free facilities for customers with disabilities, such as portable
ramps. Accessibility measures are also taken to enhance safety and convenience, including visual aids, visual alarm signals, slip
resistant tiles and handrails.
OUR CUSTOMERS
PCCW Environmental, Social and Governance Report 2021
51
CSL Mobile special offer
In May 2021, CSL Mobile offered 100GB of local mobile data for free to csl and 1O1O customers who were under
mandated quarantine at Lei Yue Mun Park and Holiday Village, Penny’s Bay Quarantine Centre, Silka Hotel in Tsuen
Wan and Dorsett Hotel, helping them stay connected with their friends and family. Coverage has been extended to
customers whose residential buildings have been identified as designated for mandatory testing. We will continue to
monitor the development and provide necessary support to our customers accordingly.
Pilot for enhancing customer onboarding experience
QR code stickers have been placed on HKT wall plates in a new
residential estate scheduled for move-in in 2022. The use of QR code
enables the new occupants to send service requests instantly via
WhatsApp for HKT’s dedicated sales team to follow up on, which
helps enhance convenience and efficiency.
OUR CUSTOMERS
PCCW Environmental, Social and Governance Report 2021
52
Planning for tomorrow’s need
It is our goal at PCCW to plan ahead and meet the needs of the future through
technological advancement.
We have the strongest 5G network in Hong Kong, with an outdoor area coverage of
99% as at the end of 2021. During the year, we have expanded 5G network coverage
to popular hiking trails and remote areas. We are also the only operator in Hong
Kong to provide seamless 5G network coverage along all MTR lines with dedicated
spectrum.
We also offer end-to-end integrated solutions to our customers and public sectors to facilitate smart healthcare, smart
properties and smart construction, assisting in their accelerated digital transformation using 5G.
Partnership with CUHK Medical Centre on 5G smart hospital
HKT has partnered with CUHK Medical Centre to transform a private hospital, making it the first in Hong Kong to
offer full 5G coverage. Leveraging the unique ultra-high speed and extremely low latency characterizing 5G, a wide
range of smart hospital applications can be supported, including remote consultation (doctor-to-doctor), remote
training (doctor-to-student) and telemedicine (doctor-to-patient). 5G also enables other innovative applications such as
the Internet of Medical Things and robotics to optimize the use of hospital resources, improving overall effectiveness
and efficiency to provide patients with the best possible care.
5G network coverage:
99%
OUR CUSTOMERS
PCCW Environmental, Social and Governance Report 2021
53
5G smart construction at Kai Tak Sports Park
HKT has joined hands with Kai Tak Sports Park Limited and Hip Hing Engineering to introduce Hong Kong’s first
construction site with a dedicated 5G network infrastructure, including 5G base stations, 5G mobile management
system and 4K high-dynamic range cameras on tower cranes. The use of 5G technology enables accelerated
transmission of large files, HD images and video footage at the construction site, enhancing the use of Building
Information Modeling and collaboration between workers. The project team is also considering the implementation
of more robotics and AI technology on site to improve occupational health and safety. Leveraging continuous
advancements in 5G applications, we hope to set a new benchmark in smart construction, making sites safer, smarter
and more efficient.
Enabling smart technologies with 5G
Since 2018, HKT and ASTRI have established a Smart City Joint Lab to drive new initiatives on smart city and 5G
development in Hong Kong. In 2021, we successfully completed a public road trial with C-V2X units installed on
traffic light poles and lamp posts along a 14 km route between Hong Kong Science Park and Sha Tin town, providing
real-time traffic intelligence to on-board units installed in test vehicles. The new C-V2X technology and its successful
trial supported the development of autonomous driving in Hong Kong, a long-term initiative of the Intelligent
Transport System in the latest Hong Kong Smart City Blueprint 2.0.
OUR CUSTOMERS
PCCW Environmental, Social and Governance Report 2021
54
We also continued to monitor electro-magnetic field emittance from our facilities, ensuring our compliance with relevant
standards on radiation safety. Similar requirements were also being extended to our suppliers. We will continue to monitor
the latest industry developments and guidelines from the Department of Health and World Health Organization to ensure the
health and safety of our customers and community.
5.3 Content Dissemination and Responsible Advertising
PCCW strives to ensure that our customers are provided with clear and accurate information when purchasing our products
and services. We comply with the Trade Descriptions Ordinance (Cap. 362) and offer guidelines and training to our sales and
marketing employees to help them fully understand our policy and compliance requirements. In 2021, there was no breach
of relevant regulations on advertising and labeling.
In terms of content dissemination to viewers from PCCW’s media platforms, our television business operations strictly
adhere to the Broadcasting Ordinance (Cap. 562) and relevant codes and guidelines. The audience is informed by on-screen
classification symbols and advisory messages before the screening of any programs with content unsuitable for children, such as
violence, strong language and nudity. For our underage audience, we provide the option of parental lock on adult-oriented
programs and offer children-friendly channels and video-on-demand content on our paid platform.
PCCW is committed to protecting the intellectual property rights of the Company, its customers and its business partners.
To ensure strict compliance with relevant laws and regulations, we have an Intellectual Property Rights Policy in place for our
staff to follow. The policy also covers our marketing materials to ensure they are free from copyright infringement.
For PCCW Solutions, the online advertising guidelines and the “Intellectual Property Rights – Infringement Claim Procedures”
are available on PCCW’s intranet for staff compliance.
5.4 Customer Service and Satisfaction
PCCW regards customer service and customer satisfaction as core indicators of service quality at our
retail operations. We constantly communicate with our customers to collect feedback and further
our understanding of their expectations. A wide range of communication channels are provided to
our customers, including service hotlines, live webchat, online enquiry, Facebook, surveys, email,
post, fax and customer service representatives in retail stores and service centers.
To provide added convenience for our customers, e-bill management and online support are available via the My HKT
platform. At the end of 2021, the platform had over 1.1 million registered accounts.
To monitor and improve the service quality of our frontline staff, we have a range of performance monitoring schemes in
place:
• Call monitoring program
• Customer transaction and net promoter score survey after calls and visits
• Mystery shopper program in retail locations – with 1,248 mystery shopper visits in 2021
We also have a set of service pledges in place. For more details, please visit our corporate website.
For any customer complaint, we target to provide a reply within two working days and resolve the case within four working
days. In 2021, over 99.8% of customer complaints were handled and resolved within four days. During the reporting period,
PCCW received 48,845 compliments and 1,724 complaints from customers.12
12 Customers from fixed-line, NETVIGATOR broadband, The Club, mobile, Now TV and ViuTV businesses
OUR CUSTOMERS
PCCW Environmental, Social and Governance Report 2021
55
In our 2021 internal customer satisfaction survey, over 95% of survey respondents rated our services in Group Strategic
Purchasing, Store & Logistics and Transport Services as “good” or “very good”. Over 84% of respondents reported that
they were “satisfied” or “very satisfied” with the service provided by Facilities Management and Portfolio Management. Our
management regularly reviews the complaints and compliments reports, customer feedback and survey results to identify
areas for improvement.
PCCW Solutions ensures that our products and services consistently meet the needs of our customers and uphold quality
excellence. Certified with ISO 9001, ISO/IEC 20000, ISO 22301, ISO/IEC 27001 standards and compliance of payment card
industry data security standard (“PCI DSS”) requirement, PCCW Solutions has established a Corporate Quality Management
System (“CQMS”) to define quality management mechanisms across all business processes, including customer service. Our
CQMS is governed by our Quality Policy and Quality Manual, led by the Top Management, Compliance and Quality Assurance
Team, and is composed of Quality Representatives from multiple teams.
PCCW Solutions has achieved the highest maturity level – Level 5 of Capability Maturity Model Integration (“CMMI”) for the
respective Application Development and Managed Services, which covers all business locations, including but not limited
to Hong Kong, mainland China, Malaysia and the Philippines. This is a recognition for PCCW Solutions’ efficiency and
effectiveness on quality service delivery that meets challenging market and customer expectations.
The Service Excellence Awards (“SEA”) is an internal scheme which aims to encourage our staff to consistently provide
excellent customer service for both external and internal customers. In 2021, a total of 120 individuals and 48 teams were
awarded the SEA with cash prizes.
Our efforts in providing excellent customer service have been recognized by different awarding organizations throughout
2021, with more than 150 accolades in various categories from the Hong Kong Customer Contact Association, Hong Kong
Management Association, Hong Kong Retail Management Association, and Mystery Shopper Service Association, among
others.
In 2021, we continued our participation in the Communications Association of Hong Kong’s Customer Complaint Settlement
Scheme, which provides mediation services to resolve disputes between customers and telecommunications service providers.
PCCW has incorporated accessibility features in its official website. We received a Silver Award under the Web Accessibility
Recognition Scheme 2021 organized by Hong Kong Internet Registration Corporation Limited.
6. OUR SUPPLY CHAIN MANAGEMENT
OUR SUPPLY CHAIN MANAGEMENT
PCCW Environmental, Social and Governance Report 2021
56
Our pledge of accelerating digital transformation would not be achieved without the help of our supply chain, which covers
a wide range of goods and services including IT, office equipment, and marketing and sales services. We extend our efforts
on sustainability to our supply chain through supplier collaboration. All our suppliers, contractors, subcontractors and service
providers are required to adopt our Supplier Code of Conduct (the “Code”), which provides a common standard for ethical
conduct and compliance requirement.
Objectives
Encourage suppliers and contractors to adopt
sustainable initiatives
Maintain stability of the
supply chain
Achieve zero bribery
and corruption
Measures
Group Purchasing Policy and Principles
Supplier Code of Conduct
Regular supplier visits and performance reviews
ISO standards for quality management system
There are growing expectations of stakeholders including the Government, customers, shareholders and employees, on
PCCW to take responsibility for its suppliers’ environmental, social and ethical practices. PCCW is increasingly making
responsible sourcing an integral part of its procurement and supply chain management processes to understand and manage
these risks.
The Group Purchasing and Supply Department (“GPS”) has formulated the Group Purchasing Policy and Principles (“GPPP”)
to include responsible business in the various processes and criteria for supplier selection and management.
Our CR Policy and ABC Policy have been established by the Group to strictly prohibit our employees from engaging in
any form of bribery or corruption at PCCW and in our supply chain. In order to ensure the effectiveness of the feedback
mechanisms, confidential channels are available to report misconduct.
6.1 Supplier Code of Conduct
To better manage the environmental and social risks along the supply chain, our suppliers are required to follow the
Code so as to ensure that the practice of our supply chain and business partners aligns with our latest codes of ethics and
professionalism. It covers issues including:
• Anti-bribery and corruption
• Conflicts of interest
• Supplier diversity
• Legal and regulatory compliance
• Human rights
• Labor standards
• Occupational safety and health
• Environmental management
OUR SUPPLY CHAIN MANAGEMENT
PCCW Environmental, Social and Governance Report 2021
57
Suppliers are required to be fully compliant with the Code in business operations, including provision of anti-corruption
policies, prevention of child or forced labor, provision of fair payment and compliant work hours, prohibition of acts of
discrimination, maintenance of freedom of association, provision of safe work conditions and management of environmental
impacts. We constantly monitor and review the Code to ensure it meets the latest laws and regulations, as well as the needs
of our business development.
6.2 Supplier Selection and Monitoring
Starting from 2017, we have been engaging an independent third party to review our approved supplier list against our
supplier engagement policy and standards. Since 2018, we have attained the ISO 9001:2015 quality management system
certification which enables us to continuously improve our procurement process and achieve the highest standard of business
practices and service offering.
Supplier performance
All new and potential suppliers are evaluated according to stringent procedures. Suppliers are required to fill in the vendor
registration form, which helps us assess various aspects including quality assurance, CSR and corporate governance. GPS
then works with an independent third party to investigate the background of the company, mainly focusing on their financial
credibility. The department also conducts an assessment of the supplier’s quality of delivery, environmental and social
compliance and internal controls.
Engagement and audits
GPS undertakes annual performance reviews of our existing major suppliers and contractors. In addition, each BU assists
in regular assessments on the performance of suppliers upon the receipt of goods and services. We have also introduced
the Selective Supplier Review and Sustainable Quarterly Review to enhance our supplier management. Any supplier with
an unsatisfactory or low rating in the review is jointly assessed by the relevant BU and GPS. When necessary, we promptly
communicate with the supplier in question on rectification or improvements. If unsatisfactory ratings are repeatedly found
or the supplier severely violates our standards, we would consider terminating the contract or blacklisting the supplier. In
2021, all existing suppliers passed the performance assessment and therefore no new entries were added to our blacklisted
suppliers list. From time to time, we conduct visits and management meetings to ensure suppliers strictly comply with our
policy requirements, and to assess their production capability and quality management system. This year, 115 supplier visits
were conducted.
OUR SUPPLY CHAIN MANAGEMENT
PCCW Environmental, Social and Governance Report 2021
58
Hong Kong
3,340
(80%)
Mainland China
280
(7%)
Regions/Countries
outside China
530
(13%)
6.3 Sustainable Procurement
Being a founding member of the Sustainable Procurement Charter launched by the Green Council, PCCW is committed to
promoting sustainable procurement practices to companies in Hong Kong. During the procurement process, social, ethical
and environmental performance factors are incorporated into our consideration.
Sustainable procurement measures are also included in GPS’s Risks and Opportunities Register. These include promoting
sustainable procurement concepts to our staff, arranging for staff to attend related seminars and training, and specifying
environmental and social expectations and requirements on suppliers in the Code. Furthermore, we clearly state the clauses
of “Environmental Protection”, “Notes for Sellers for CSR” and “Energy Efficiency”, among others, in our procurement
contracts to ensure effective demonstration of our commitment.
In addition, we are aware of our environmental and social impact throughout the supply chain. We worked with over 4,150
suppliers during the reporting year, of which 80% are based in Hong Kong. The procurement decision of prioritizing local
suppliers helps minimize carbon emissions resulting from transportation.
Distribution of suppliers by geographical locations
ASSURANCE REPORT
ASSURANCE REPORT
PCCW Environmental, Social and Governance Report 2021
59
VERIFICATION STATEMENT
Scope of Verification
Hong Kong Quality Assurance Agency (“HKQAA”) has been commissioned by PCCW Limited (“PCCW”) (SEHK: 0008) to
undertake an independent verification for its Environmental, Social and Governance Report 2021 (“The Report”).
The scope of HKQAA’s verification covers the data and information associating to PCCW’s sustainability performance as
described in the Report for the period of January 1, 2021 to December 31, 2021.
Level of Assurance and Methodology
The process applied in this verification was referring to the International Standard on Assurance Engagements 3000 (Revised)
– Assurance Engagements Other Than Audits or Reviews of Historical Financial Information issued by the International
Auditing and Assurance Standards Board. Our evidence gathering process was designed to obtain a reasonable level of
assurance for devising the verification conclusion. The extent of this verification process undertaken was provided for the
criteria set in The Environmental, Social and Governance Reporting Guide (“ESG Reporting Guide”) to the Rules Governing
the Listing of Securities on The Stock Exchange of Hong Kong Limited.
The systems and processes for collecting, collating and reporting the environmental performance data were verified. Our
verification procedure covered reviewing of relevant documentation, interviewing responsible personnel with accountability
for preparing the Report and verifying the raw data and supporting evidence of the selected samples during the verification
process.
Independence
PCCW is responsible for the collection and presentation of the information presented. HKQAA does not involve in calculating,
compiling, or in the development of the Report. Our verification activities are independent from PCCW.
Conclusion
On the basis of our verification results and in accordance with the verification procedures undertaken, it is the opinion of the
HKQAA’s verification team that:
• The Report has complied with all mandatory disclosure requirements and “comply or explain” provisions outlined in the
ESG Reporting Guide;
• The Report illustrates PCCW’s sustainability performance in a balanced, comparable, clear and timely manner; and
• The data and information stated in the Report are reliable and complete.
The Report reflects appropriately PCCW’s context and materiality of its sustainability issues and allows stakeholders to have a
clear understanding of its commitments and stewardship towards sustainability management.
Signed on behalf of Hong Kong Quality Assurance Agency
Connie Sham
Head of Audit
March 2022
EXTERNAL CHARTERS AND MEMBERSHIP
EXTERNAL CHARTERS AND MEMBERSHIP
PCCW Environmental, Social and Governance Report 2021
60
External Charters
Name of Association Name of Charter
Environment Bureau Charter on External Lighting
Environment Bureau Energy Saving Charter
Department of Health/
Occupational Safety & Health Council
Joyful@Healthy Workplace Charter
Department of Health Organ Donation Promotion Charter
Environmental Protection Department Friends of EcoPark
Equal Opportunities Commission The Racial Diversity & Inclusion Charter for Employers
Green Council Sustainable Procurement Charter
Labour Department/Occupational Safety & Health Council Occupational Safety Charter
Occupational Safety & Health Council Charter on Preferential Appointment of OSH Star Enterprise
Membership
Name of Association Type of Membership
Business Environment Council Council Member
Employers’ Federation of Hong Kong Founding & Council Member
Girls Go Tech, The Women’s Foundation Technology Partner, Council Member
Food Grace Green Membership
The Green Earth Green Earth Companion, Water Category
Hong Kong Management Association Charter Member
The Hong Kong Council of Social Service Caring Company Patron’s Club – Coral Membership
The Hong Kong Institute of Human Resource
Management
Corporate Member
PCCW Environmental, Social and Governance Report 2021
PERFORMANCE DATA SUMMARY
PERFORMANCE DATA SUMMARY
61
Environmental Performance Data
2019 2020 2021
2019 2020 2021 Change
(2021 vs 2020)
Types of emissions and respective emissions data13
Sulfur oxides (“SOx”) – Direct (kg) 6.11 6.03 5.97 -1.14%
Nitrogen oxides (“NOx”) – Direct (kg) 3,221 2,950 2,775 -5.92%
Particulate matter (“PM”) – Direct (kg) 294.42 265.41 252.11 -5.01%
GHG emissions and intensity14
GHG emissions – Scope 115 (tonnes CO2
e) 6,953 7,359 6,548 -11.01%
GHG emissions – Scope 216 (tonnes CO2
e) 236,018 229,092 186,406 -18.63%
GHG emissions – Scope 317 (tonnes CO2
e)
– Paper consumption 1,115.60 1,039.12 1,023.07 -1.54%
– Water consumption and sewage
discharge18 206.63 213.25 225.82 +5.90%
Total GHG emissions (Scopes 1+2+3)
(tonnes CO2
e) 244,293 237,703 194,203 -18.30%
GHG emissions intensity per employee19
(tonnes CO2
e/employee) 16.28 17.00 15.06 -11.41%
GHG emissions intensity per million
revenue20 (tonnes CO2
e/HK$ million) 6.69 6.61 5.02 -23.94%
Hazardous waste produced
Solid waste21 (tonnes) 202.66 549.90 246.85 -55.11%
WEEE disposal22
– Electronic and IT equipment (pieces) 66,227 47,021 68,280 +45.21%
– Equipment cables (meters) 178,134 127,790 209,417 +63.88%
13 Air emissions are generated from petrol and diesel fuel combustion in vehicles. The emission factors are adopted from “How to prepare an ESG Report –
Appendix 2: Reporting Guidance on Environmental KPIs” published by HKEX. 14 GHG emissions are calculated based on “Guidelines to Account for and Report on Greenhouse Gas Emissions and Removals for Buildings (Commercial,
Residential or Institutional Purposes) in Hong Kong (2010 Edition)” published by the Environmental Protection Department (“EPD”) and the Electrical and
Mechanical Services Department (“EMSD”) of the Hong Kong SAR Government, unless otherwise stated in the following notes. 15 Scope 1 emissions comprise HFC and PFC emissions from the use of refrigerants and emissions from our standby emergency generators and vehicle fleet that
run on diesel as well as our other vehicle fleet that runs on petrol. The global warming potentials used for calculation are adopted from Intergovernmental
Panel on Climate Change (“IPCC”) Fifth Assessment Report. 16 Scope 2 emissions are generated from the electricity consumed by PCCW’s major operations with individual meters. Emissions factors are adopted from the
latest sustainability reports of local power companies. 17 The figures on Scope 3 emissions are generated from office paper consumption, water consumption and sewage discharge. 18 The emission factors for fresh water processing and sewage processing are adopted from the latest annual report of the Water Supplies Department and the
sustainability report of the Drainage Services Department. 19 As of December 31, 2021, the number of employees in Hong Kong was 12,893, which is also the basis for electricity, energy and water intensity calculations. 20 The calculation is based on the revenue of PCCW, which was HK$38,654 million in 2021. This figure is also the basis for electricity, energy and water
intensity calculations. The revenue in 2020 was adjusted, and the intensity calculations in 2020 were restated correspondingly. 21 Solid waste includes industrial batteries (valve-regulated lead-acid battery), office batteries and fluorescent tubes. The increase in 2020 was due to the
Group’s disposing and recycling of a large quantity of waste industrial batteries (545.57 tonnes) that reached the end of their life cycle; these waste
industrial batteries were recycled. In 2021, the waste industrial batteries being recycled was reduced to 244.01 tonnes. 22 The figure is reported on a Group basis comprising HKT and parent company PCCW. It does not include WEEE disposed of by the Group on behalf of
customers. The major increment in 2021 was the result of disposal of aged and end-of-life equipment (including voice switching unit and broadband
transmission equipment).
PCCW Environmental, Social and Governance Report 2021
PERFORMANCE DATA SUMMARY
62
2019 2020 2021
2019 2020 2021 Change
(2021 vs 2020)
Non-hazardous waste produced
General waste23 (tonnes) 849.25 766.61 824.47 +7.55%
Construction waste24 (tonnes) 140.25 171.80 742.79 +332.36%
Waste management and results
Scrap materials recycled25 (pieces) 230,145 191,026 243,713 +27.58%
Toner and ink cartridges recycled (pieces) 1,808 1,454 1,152 -20.77%
Paper recycled26 (tonnes) 142.20 107.06 118.48 +10.67%
Scrap metals recycled27 (tonnes) 26.38 26.24 11.63 -55.68%
Direct and/or indirect energy consumption by type and intensity
Electricity (kWh) 383,144,892 371,067,770 358,192,525 -3.47%
Electricity intensity per employee
(GJ/employee) 91.91 95.55 100.01 +4.68%
Electricity intensity per million revenue
(GJ/HK$ million) 37.78 37.12 33.36 -10.14%
Petrol fuel – vehicle fleet (L) 116,493 108,031 97,912 -9.37%
Diesel fuel – vehicle fleet (L) 270,578 274,599 281,043 +2.35%
Diesel fuel – standby emergency
generators (L) 49,260 73,847 32,335 -56.21%
Total energy consumption (GJ) 1,394,866 1,352,146 1,304,983 -3.49%
Energy intensity28 per employee
(GJ/employee) 92.94 96.71 101.22 +4.66%
Energy intensity per million revenue
(GJ/HK$ million) 38.21 37.12 33.36 -10.14%
Water consumption and intensity
Water consumption29 (m3
) 331,665 341,744 366,000 +7.10%
Water intensity per employee (m3
/employee) 22.10 24.44 28.39 +16.14%
Water intensity per million revenue
(m3
/HK$ million) 9.09 9.50 9.47 -0.30%
Total packaging material
Shopping bags (tonnes) 15.97 18.60 16.16 -13.12%
23 General waste mainly includes general office waste. 24 Construction waste increased due to more store closures and store renovations as compared to 2020. In addition, three exchange buildings have undergone
major re-roofing in 2021 while only minor maintenance was conducted in 2020. 25 Scrap materials such as scrap cables, scrap telephones, obsolete devices and accessories, modems and routers, set-top boxes, WEEE and transmission
equipment. 26 Paper recycling data in 2019 and 2020 have been restated after data review. 27 Scrap metals include copper, metal and steel. 28 The calculation of energy intensity includes consumption of electricity, petrol fuel and diesel fuel. 29 Water consumption of PCCW’s major operations with individual meters.
PCCW Environmental, Social and Governance Report 2021
REFERENCES TO HKEX ESG
REPORTING GUIDE
REFERENCES TO HKEX ESG REPORTING GUIDE
63
A. Environmental
Aspect A1: Emissions PCCW’s Comments
General Disclosure Information on:
(a) the policies; and
(b) compliance with relevant laws and
regulations that have a significant impact on
the issuer
relating to air and greenhouse gas emissions,
discharges into water and land, and generation
of hazardous and non-hazardous waste.
3. Our Environment
KPI A1.1 The types of emissions and respective emissions
data.
Performance Data Summary
KPI A1.2 Direct (Scope 1) and energy indirect (Scope 2)
greenhouse gas emissions (in tonnes) and,
where appropriate, intensity.
Performance Data Summary
KPI A1.3 Total hazardous waste produced (in tonnes) and,
where appropriate, intensity.
Performance Data Summary
KPI A1.4 Total non-hazardous waste produced (in tonnes)
and, where appropriate, intensity.
Performance Data Summary
KPI A1.5 Description of emission target(s) set and steps
taken to achieve them.
3. Our Environment
• 3.1 Climate Change, Energy
Consumption and GHG Emissions
• 3.3 Environmental Targets
• 3.4 Building a Smart City
• 3.5 Employee Environmental Awareness
KPI A1.6 Description of how hazardous and
non-hazardous wastes are handled, and a
description of reduction target(s) set and steps
taken to achieve them.
3. Our Environment
• 3.2 Sustainable Use of Resources
• 3.3 Environmental Targets
• 3.5 Employee Environmental Awareness
PCCW Environmental, Social and Governance Report 2021
REFERENCES TO HKEX ESG REPORTING GUIDE
64
A. Environmental
Aspect A2: Use of Resources PCCW’s Comments
General Disclosure Policies on the efficient use of resources,
including energy, water and other raw materials.
3. Our Environment
KPI A2.1 Direct and/or indirect energy consumption by
type in total (kWh in ’000s) and intensity.
Performance Data Summary
KPI A2.2 Water consumption in total and intensity. Performance Data Summary
KPI A2.3 Description of energy use efficiency target(s) set
and steps taken to achieve them.
3. Our Environment
• 3.1 Climate Change, Energy
Consumption and GHG Emissions
• 3.3 Environmental Targets
• 3.4 Building a Smart City
• 3.5 Employee Environmental Awareness
KPI A2.4 Description of whether there is any issue in
sourcing water that is fit for purpose, water
efficiency target(s) set and steps taken to achieve
them.
PCCW’s operation is not located in waterstressed regions and does not involve intensive
water use. We reuse waste water from watercooled condensers. For details, please refer to:
3. Our Environment
• 3.2 Sustainable Use of Resources
• 3.3 Environmental Targets
KPI A2.5 Total packaging material used for finished
products (in tonnes) and, if applicable, with
reference to per unit produced.
Performance Data Summary
PCCW Environmental, Social and Governance Report 2021
REFERENCES TO HKEX ESG REPORTING GUIDE
65
A. Environmental
Aspect A3: The Environment and Natural Resources PCCW’s Comments
General Disclosure Policies on minimising the issuer’s significant
impact on the environment and natural
resources.
3. Our Environment
KPI A3.1 Description of the significant impacts of activities
on the environment and natural resources and
the actions taken to manage them.
3. Our Environment
Aspect A4: Climate Change
General Disclosure Policies on identification and mitigation of
significant climate-related issues which have
impacted, and those which may impact, the
issuer.
3. Our Environment
KPI A4.1 Description of the significant climate-related
issues which have impacted, and those which
may impact, the issuer, and the actions taken to
manage them.
3. Our Environment
• 3.1 Climate Change, Energy
Consumption and GHG Emissions
PCCW Environmental, Social and Governance Report 2021
REFERENCES TO HKEX ESG REPORTING GUIDE
66
B. Social
Employment and Labour Practices
Aspect B1: Employment PCCW’s Comments
General Disclosure Information on:
(a) the policies; and
(b) compliance with relevant laws and
regulations that have a significant impact on
the issuer
relating to compensation and dismissal,
recruitment and promotion, working hours,
rest periods, equal opportunity, diversity, antidiscrimination, and other benefits and welfare.
2. Our People
KPI B1.1 Total workforce by gender, employment type,
age group and geographical region.
2. Our People
• 2.4 Diversity and Inclusiveness
KPI B1.2 Employee turnover rate by gender, age group
and geographical region.
2. Our People
• 2.3 Talent Retention and Development
Aspect B2: Health and Safety
General Disclosure Information on:
(a) the policies; and
(b) compliance with relevant laws and
regulations that have a significant impact on
the issuer
relating to providing a safe working environment
and protecting employees from occupational
hazards.
2. Our People
KPI B2.1 Number and rate of work-related fatalities
occurred in each of the past three years
including the reporting year.
2. Our People
• 2.2 Occupational Safety and Health
KPI B2.2 Lost days due to work injury. 2. Our People
• 2.2 Occupational Safety and Health
KPI B2.3 Description of occupational health and safety
measures adopted, how they are implemented
and monitored.
2. Our People
• 2.1 Employee Well-being
• 2.2 Occupational Safety and Health
PCCW Environmental, Social and Governance Report 2021
REFERENCES TO HKEX ESG REPORTING GUIDE
67
B. Social
Aspect B3: Development and Training PCCW’s Comments
General Disclosure Policies on improving employees’ knowledge and
skills for discharging duties at work. Description
of training activities.
2. Our People
• 2.3 Talent Retention and Development
KPI B3.1 The percentage of employees trained by gender
and employee category.
2. Our People
• 2.3 Talent Retention and Development
KPI B3.2 The average training hours completed per
employee by gender and employee category.
2. Our People
• 2.3 Talent Retention and Development
Aspect B4: Labour Standards
General Disclosure Information on:
(a) the policies; and
(b) compliance with relevant laws and
regulations that have a significant impact on
the issuer
relating to preventing child and forced labour.
2. Our People
KPI B4.1 Description of measures to review employment
practices to avoid child and forced labour.
PCCW respects and upholds fundamental
human rights. We prohibit forced labor and child
labor across our operations. For details, please
refer to:
2. Our People
KPI B4.2 Description of steps taken to eliminate such
practices when discovered.
PCCW Environmental, Social and Governance Report 2021
REFERENCES TO HKEX ESG REPORTING GUIDE
68
B. Social
Operating Practices
Aspect B5: Supply Chain Management PCCW’s Comments
General Disclosure Policies on managing environmental and social
risks of the supply chain.
6. Our Supply Chain Management
KPI B5.1 Number of suppliers by geographical region. 6. Our Supply Chain Management
• 6.3 Sustainable Procurement
KPI B5.2 Description of practices relating to engaging
suppliers, number of suppliers where the
practices are being implemented, how they are
implemented and monitored.
6. Our Supply Chain Management
• 6.1 Supplier Code of Conduct
• 6.2 Supplier Selection and Monitoring
• 6.3 Sustainable Procurement
KPI B5.3 Description of practices used to identify
environmental and social risks along the supply
chain, and how they are implemented and
monitored.
6. Our Supply Chain Management
• 6.1 Supplier Code of Conduct
• 6.2 Supplier Selection and Monitoring
KPI B5.4 Description of practices used to promote
environmentally preferable products and services
when selecting suppliers, and how they are
implemented and monitored.
6. Our Supply Chain Management
• 6.3 Sustainable Procurement
PCCW Environmental, Social and Governance Report 2021
REFERENCES TO HKEX ESG REPORTING GUIDE
69
B. Social
Aspect B6: Product Responsibility PCCW’s Comments
General Disclosure Information on:
(a) the policies; and
(b) compliance with relevant laws and
regulations that have a significant impact on
the issuer
relating to health and safety, advertising,
labelling and privacy matters relating to products
and services provided and methods of redress.
5. Our Customers
KPI B6.1 Percentage of total products sold or shipped
subject to recalls for safety and health reasons.
It is not material to PCCW’s business.
KPI B6.2 Number of products and service related
complaints received and how they are dealt
with.
5. Our Customers
• 5.4 Customer Service and Satisfaction
KPI B6.3 Description of practices relating to observing
and protecting intellectual property rights.
5. Our Customers
• 5.3 Content Dissemination and
Responsible Advertising
KPI B6.4 Description of quality assurance process and
recall procedures.
5. Our Customers
• 5.2 Reliable and Responsible Services and
Products
• 5.4 Customer Service and Satisfaction
KPI B6.5 Description of consumer data protection and
privacy policies, how they are implemented and
monitored.
5. Our Customers
• 5.1 Customer Data Privacy and Security
PCCW Environmental, Social and Governance Report 2021
REFERENCES TO HKEX ESG REPORTING GUIDE
70
B. Social
Aspect B7: Anti-corruption PCCW’s Comments
General Disclosure Information on:
(a) the policies; and
(b) compliance with relevant laws and
regulations that have a significant impact on
the issuer
relating to bribery, extortion, fraud and money
laundering.
1. Corporate Social Responsibility at PCCW
• 1.3 Ethics and Integrity
KPI B7.1 Number of concluded legal cases regarding
corrupt practices brought against the issuer or
its employees during the reporting period and
the outcomes of the cases.
1. Corporate Social Responsibility at PCCW
• 1.3 Ethics and Integrity
KPI B7.2 Description of preventive measures and
whistleblowing procedures, how they are
implemented and monitored.
1. Corporate Social Responsibility at PCCW
• 1.3 Ethics and Integrity
KPI B7.3 Description of anti-corruption training provided
to directors and staff.
1. Corporate Social Responsibility at PCCW
• 1.3 Ethics and Integrity
B. Social
Community
Aspect B8: Community Investment PCCW’s Comments
General Disclosure Policies on community engagement to
understand the needs of the communities where
the issuer operates and to ensure its activities
take into consideration the communities’
interests.
4. Our Community
KPI B8.1 Focus areas of contribution. 4. Our Community
KPI B8.2 Resources contributed to the focus area. 4. Our Community

'''
class ESG():
    model = 0
    tokenizer = 0
    checkpoint = 0
    trainer = 0
    example_to_features = 0
    start_logits = 0
    end_logits = 0
    n_best = 20
    max_answer_length = 30
    top_n = 20
    # predicted_answers = []
    
    def __init__(self, checkpoint):
        self.checkpoint = checkpoint
        self.model = AutoModelForQuestionAnswering.from_pretrained(checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.args = TrainingArguments(
            "main_test",
            evaluation_strategy="no",
            save_strategy="epoch",
            learning_rate=2e-5,
            num_train_epochs=3,
            weight_decay=0.01,
            per_device_train_batch_size=6,
            per_device_eval_batch_size=128,
            # fp16=True,
            no_cuda=True,
            push_to_hub=False,
            save_total_limit=1
        )
        self.trainer = Trainer(
            model=self.model,
            args=self.args,
            # train_dataset=train_dataset,
            # eval_dataset=eval_set,
            tokenizer=self.tokenizer,

        )
        print('[model init success]')

    def update_model(self, checkpoint):
        print('=============[update_model]==============')
        self.model = AutoModelForQuestionAnswering.from_pretrained(checkpoint)
        self.trainer = Trainer(
            model=self.model,
            args=self.args,
            # train_dataset=train_dataset,
            # eval_dataset=eval_set,
            tokenizer=self.tokenizer,

        )
        
    def get_model_result(self, example):
        print('=============[get_model_result]==============')
        example_id = example["id"]
        context = example["context"]
        answers = []
        for feature_index in self.example_to_features[example_id]:
            # print((feature_index,tokenizer.decode(eval_set["input_ids"][feature_index])))
            start_logit = self.start_logits[feature_index]
            end_logit = self.end_logits[feature_index]
            offsets = eval_set["offset_mapping"][feature_index]

            start_indexes = np.argsort(
                start_logit)[-1: -self.n_best - 1: -1].tolist()
            end_indexes = np.argsort(end_logit)[-1: -self.n_best - 1: -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length.
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > self.max_answer_length
                    ):
                        continue
                    ans = context[offsets[start_index][0]: offsets[end_index][1]]
                    if re.search(r'\d', ans):
                        answers.append(
                            {
                                # "id": len(answers),
                                "answer": ans,
                                "context": tokenizer.decode(eval_set["input_ids"][feature_index], skip_special_tokens=False),
                                "logit_score": start_logit[start_index] + end_logit[end_index],
                                # "origin_text": tokenizer.decode(eval_set["input_ids"][feature_index])
                            }
                        )

                    # print((context[offsets[start_index][0] : offsets[end_index][1]], start_logit[start_index] + end_logit[end_index]))

        # best_answer = max(answers, key=lambda x: x["logit_score"])
        top_20_answers = heapq.nlargest(
            self.top_n, answers, key=lambda s: s['logit_score'])
        # predicted_answers.append(
        #     {"id": example_id, "prediction_text": best_answer["text"]})
        example['answer_set'] = top_20_answers
        return example
        # print(top_20_answers)
        
    def get_fiture_set(self, small_eval_set, eval_set, top_n=20):
        print('=============[get_fiture_set]==============')
        self.top_n = top_n
        predictions, _, _ = self.trainer.predict(eval_set)
        self.start_logits, self.end_logits = predictions
        self.example_to_features = collections.defaultdict(list)
        for idx, feature in enumerate(eval_set):
            self.example_to_features[feature["example_id"]].append(idx)
        
        firuge_set = small_eval_set.map(self.get_model_result, batched=False)
        return firuge_set
        
        


model_checkpoint = "bert-finetuned-esgQA-fine-grained-8-2"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

max_length = 384
stride = 128

id = 0


def add_question_to_get_fiture(example):
    global id
    id = id + 1
    example['question'] = "How much {}?".format(example['path'])
    example['context'] = example['text']
    # example['answers'] = {
    #     'text': [example['value']],
    #     'answer_start': [example['text'].find(example['value'])]
    # }
    example['id'] = str(id)
    return example


def add_question(example, template):
    answer_list = example['answer_set']
    format_answer_list = []
    for answer in answer_list:
        if re.search(r'\d', answer['text']):
            format_answer_list.append(answer['text'])

    global id
    id = id + 1
    example['question'] = "How much {}?".format(example['path'])
    example['context'] = example['text']
    # example['answers'] = {
    #     'text': [example['value']],
    #     'answer_start': [example['text'].find(example['value'])]
    # }
    example['id'] = str(id)
    return example


def preprocess_validation_examples(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids
    return inputs



# input = input.replace('\n',' ')

# input = '''
# The Group is principally engaged in education services. No substantial emissions are produced by
# combustion of any fuels in daily operation as the Group is not engaged in any industrial production.
# During the Reporting Period, the principal type of emission of the Group is exhaust generated by the
# Group’s self-owned vehicles. The main emission data are as follows:
# Major emissions Unit
# Emission
# volume
# Nitrogen oxide (NOx) Gram 673,012.0
# Sulphur dioxide (SOx) Gram 637.2
# Particulate Matter Gram 66,032.1
# '''

def get_question_set(example, template):
    answer_set = example['answer_set'][0]
    id = 0
    format_answer_set = {
        'question':[],
        'context':[],
        'id':[]
    }
    # print(answer_set)
    for answer in answer_set:
        # print(answer)
        id = id + 1
        format_answer_set['id'].append(str(id))
        format_answer_set['question'].append(template.format(answer['answer']))
        # print(answer['context'].split('[SEP]'))
        format_answer_set['context'].append(answer['context'].split('[SEP]')[1])

    return Dataset.from_dict(format_answer_set)

def question(eval_firuge_set, template):
    next_question_set = get_question_set(eval_firuge_set, template)

    eval_set = next_question_set.map(
        preprocess_validation_examples,
        batched=True,
        remove_columns=next_question_set.column_names,
    )
    print((len(next_question_set),len(eval_set)))

    result_set = extractor.get_fiture_set(next_question_set, eval_set, 5)
    return result_set

extractor = ESG(model_checkpoint)
# get original data
example = {
    'text': [input],
    'path': ['CO2EquivalentsEmissionTotal'],
}
# for i in range(10):
#     example['text'].append(input)
#     example['path'].append('emission')

test_data = Dataset.from_dict(example)
small_eval_set = test_data.map(
    add_question_to_get_fiture, remove_columns=test_data.column_names)

eval_set = small_eval_set.map(
    preprocess_validation_examples,
    batched=True,
    remove_columns=small_eval_set.column_names,
)
print((len(small_eval_set),len(eval_set)))

eval_firuge_set = extractor.get_fiture_set(small_eval_set, eval_set, 5)
eval_firuge_set.to_csv('data/figure.csv',index=False)

max_length = max_length + 50
extractor.update_model('bert-large-uncased-whole-word-masking-finetuned-squad')





next_question_set = get_question_set(eval_firuge_set, 'What is the data {} about?')
eval_set = next_question_set.map(
    preprocess_validation_examples,
    batched=True,
    remove_columns=next_question_set.column_names,
)
print((len(next_question_set),len(eval_set)))
result_set = extractor.get_fiture_set(next_question_set, eval_set, 5)
result_set.to_csv('data/figure_about.csv',index=False)


next_question_set = get_question_set(eval_firuge_set, 'What is the unit of {}?')
eval_set = next_question_set.map(
    preprocess_validation_examples,
    batched=True,
    remove_columns=next_question_set.column_names,
)
print((len(next_question_set),len(eval_set)))
result_set = extractor.get_fiture_set(next_question_set, eval_set, 5)
result_set.to_csv('data/figure_unit.csv',index=False)


next_question_set = get_question_set(eval_firuge_set, 'What year is {} about?')
eval_set = next_question_set.map(
    preprocess_validation_examples,
    batched=True,
    remove_columns=next_question_set.column_names,
)
print((len(next_question_set),len(eval_set)))
result_set = extractor.get_fiture_set(next_question_set, eval_set, 5)
result_set.to_csv('data/figure_year.csv',index=False)




# result_set = question(eval_firuge_set,'What is the data {} about?')
# result_set.to_csv('data/figure_about.csv',index=False)
# # next question What is the unit of?
# result_set = question('What is the unit of {}?')
# result_set.to_csv('data/figure_about.csv',index=False)
# # next question What year?
# result_set = question('What year is {} about?')
# result_set.to_csv('data/figure_about.csv',index=False)