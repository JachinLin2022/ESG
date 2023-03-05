import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
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
2021 ESG REPORT
| Brookfield Asset Management
1
2021 ESG REPORT
Brookfield Asset
Management
Contents
INTRODUCTION 3
Brookfield at a Glance .......................................................4
Brookfield Around the World ..........................................5
2021 Sustainability Highlights .........................................6
Letter to Stakeholders.......................................................7
Who We Are .........................................................................9
ESG AT BROOKFIELD 10
Our Guiding ESG Principles ...........................................11
ESG Affiliations and Partnerships.................................12
Stakeholder Engagement ..............................................14
About This Report.............................................................15
Comprehensive Topic Review and Analysis ...............16
OUR INVESTMENT APPROACH 18
ESG Integration into Our Investment Process ..........19
Proxy Voting.......................................................................20
Systemic Risk Management ...........................................21
Sustainable Finance.........................................................22
BUILDING A BETTER WORLD 23
An Interview With Mark Carney and
Connor Teskey about the Transition to Net Zero.....24
Climate Change Strategy ................................................26
Water and Waste ..............................................................30
OUR PEOPLE 32
Human Capital Development........................................33
Diversity, Equity and Inclusion.......................................36
Occupational Health and Safety ...................................39
GOVERNANCE 41
Corporate Governance and Ethics...............................42
Brookfield Asset Management
Board of Directors............................................................43
ESG Governance...............................................................44
Business Ethics..................................................................45
Human Rights and Modern Slavery .............................46
Responsible Contracting ................................................48
Audit Oversight..................................................................49
Executive Compensation................................................49
Data Privacy and Security ...............................................50
KPI APPENDICES 51
Comprehensive Topic Review and Analysis ...............52
Key Performance Metrics ...............................................54
GRI Content Index ............................................................56
SASB Index .........................................................................61
2021 ESG REPORT | Brookfield Asset Management 3
Introduction
Brookfield at a Glance
Brookfield Around the World
2021 Sustainability Highlights
Letter to Stakeholders
Who We Are
4
INTRODUCTION
2021 ESG REPORT | Brookfield Asset Management
Brookfield at a Glance
Brookfield Asset Management is a premier global asset manager with $720 billion in assets under management (AUM).[1] [2] Our business philosophy is based on our conviction
that acting responsibly toward our stakeholders is foundational to operating a productive, profitable and sustainable business, and that value creation and sustainable
development are complementary goals. This view has been underpinned by Brookfield's 100-plus year history as an owner and operator of long-term assets that help form the
backbone of the global economy.
RENEWABLE POWER
& TRANSITION
$68B
AUM
Hydro
Wind
Energy Transition[3]
Solar
INFRASTRUCTURE
$140B
AUM
Renewable Power
Transport
Utilities
Data
Midstream
PRIVATE EQUITY
$105B
AUM
Industrials
Business Services
Infrastructure Services
Residential & Directly
Held Investments
REAL ESTATE
$256B
AUM
Multifamily
Logistics
Office
Retail Hospitality
Alternative Sectors[4]
CREDIT & INSURANCE
SOLUTIONS
$151B
AUM
Performing Credit
Opportunistic Credit
Direct Lending
1 “Brookfield,” the “Firm,” “we,” “us” or “our” refers to Brookfield Asset Management Inc. and includes its activities undertaken as an asset manager for private funds, public
securities, public issuers (Brookfield Renewable Partners L.P., Brookfield Infrastructure Partners L.P.) and its other investment programs.
2 Oaktree Capital manages credit strategies and operates independently from Brookfield. Accordingly, this report does not cover the activities undertaken by Oaktree Capital.
For information about Oaktree’s ESG approach and initiatives, visit www.oaktreecapital.com. Total AUM also includes $14 billion from our Public Securities Group and
$7 billion from our Reinsurance Group. AUM figures as of March 31, 2022.
3 Energy transition includes distributed generation, storage and other.
4 Alternative sectors include triple net lease, manufactured housing, student housing, senior living, life science and single family properties.
5
INTRODUCTION
2021 ESG REPORT | Brookfield Asset Management
Brookfield Around the World
NORTH AMERICA
$432B AUM
~69,500
Operating Employees
SOUTH AMERICA
$50B AUM
~28,200
Operating Employees
EUROPE &
MIDDLE EAST
$135B AUM
~36,500
Operating Employees
ASIA PACIFIC
$103B AUM
~43,100
Operating Employees
SCALE OF ORGANIZATION
30+
Countries
~2,300
Asset Management Employees
~180,000
Operating Employees
For more information, please visit our website at www.brookfield.com.
6
INTRODUCTION
2021 ESG REPORT | Brookfield Asset Management
2021 Sustainability Highlights
Net zero by 2050
commitment to the goal of reaching net-zero emissions by 2050
or sooner across all assets under management
~2/3
set our 2030 net-zero interim target to reduce Scope 1 and 2 emissions
across $147 billion of our AUM by approximately two-thirds
$15 billion
raised for our inaugural transition Fund,
the Brookfield Global Transition Fund
$8 billion
in issuances across green bonds, hybrid
securities and sustainability-linked debt
and loans
~29 million
metric tons of CO2
 net emissions equivalent
(mtCO2
e) avoided through Brookfield’s
renewable power generation
170%
increase in female representation at the managing
partner/managing director levels over the last five years
39%
of our employee population is comprised
of underrepresented ethnicities
7
INTRODUCTION
2021 ESG REPORT | Brookfield Asset Management
Letter to Stakeholders
At our core, Brookfield invests in the places where people live and work, in the ways
they transport themselves and their goods, and in how they power their lives. That
means sustainability is critical to what we do. Sound environmental, social and
governance (ESG) practices are essential to building resilient assets and businesses—
while also creating long-term value for our investors and stakeholders. And in 2021, we
took big strides forward in several areas of ESG—particularly in addressing climate
change and the environment.
WE MADE SUBSTANTIAL PROGRESS ON
OUR CLIMATE-RELATED GOALS
We believe that we can make a significant contribution
to the global decarbonization effort. That means
looking at net zero from two perspectives: within
Brookfield, through our commitment to reach the goal
of net-zero greenhouse gas (GHG) emissions by 2050
or sooner in our own operations; and externally,
through our transition investing and sustainable
finance initiatives. In both cases, we are well positioned
to provide strong support to the transition to net zero,
based on our global reach, access to large-scale
capital, operating expertise—particularly in
decarbonization technologies—and deep experience
developing and managing clean energy assets.
In becoming a signatory to the Net Zero Asset
Managers (NZAM) initiative in 2021, we pledged to:
 Work on decarbonization goals consistent with the
aim of reaching net-zero emissions by 2050 or
sooner across all our assets under management;
 Set an interim target of a specific percentage of our
assets to be managed in line with net zero, with
targeted emissions reduction by 2030; and
 Review this target at least every five years, with a
view to increasing the share of assets under
management (AUM) covered until 100% of assets
are included.
As a key step, we have set an interim target of reducing
Scope 1 and 2 emissions across $147 billion of our
AUM by approximately two-thirds by 2030. Meeting
this target—which encompasses our renewable
power, infrastructure, private equity and real estate
businesses—will reduce emissions across these
assets by roughly one million metric tons of carbon
dioxide equivalent.
We are also in the midst of developing actionable
decarbonization plans for our remaining AUM, and
intend to substantially increase the share of our inscope assets over time.
The global transition to net zero will require large
economic adjustments, the potential rewiring of
virtually every industry and significant capital
investment. In 2021, we launched the Brookfield Global
Transition Fund (BGTF) to invest in and help facilitate
that transition. We have been thrilled with the market’s
reception of BGTF, which raised $15 billion from
investors—including a $2 billion commitment from
Brookfield—and we have already started putting the
Fund’s capital to work in investments across the
decarbonization spectrum globally. 
8
INTRODUCTION
2021 ESG REPORT | Brookfield Asset Management
Addressing the issue of climate change, and deciding
how it should impact investment decisions, is a top
agenda item for every chief investment officer allocating
capital today. While everything does not have to
become green immediately, every business does need
to begin the shift to a cleaner future. As Mark Carney,
our Head of Transition Investing, has said: “The net-zero
transition doesn’t mean flipping a green switch or
investing only in companies that are already green.
Transition means transition.” Not only are we guiding
significant capital to these opportunities, but we are
also bringing our deep operational expertise to bear.
We published the Brookfield Asset Management
Green Bond and Preferred Securities Framework,
which lays out the process for selecting eligible
investments, the use and management of proceeds,
and the reporting frequency and format. In addition,
we stepped up our activity in the sustainable finance
market. Our issuance of green bonds, hybrid securities
and sustainability-linked debt and loans more than
doubled, to $8 billion in 2021 from $3.6 billion in 2020.
Many of our assets and investments are well suited for
sustainable financing, and we continue to look for
opportunities to access capital in this manner.
In 2021, we became supporters of the Task Force on
Climate-Related Financial Disclosures (TCFD), and we
are aligning our practices with the organization’s
recommendations. That includes embedding climate
change considerations even further into our business
strategy through our net-zero commitment, our
sustainable financing efforts and the launch of BGTF.
We plan to publish fiscal-year TCFD disclosures for our
listed renewable power affiliate, Brookfield Renewable
Partners, this year, and for Brookfield Asset
Management in 2023.
We also undertook a climate risk assessment last year
to better understand the potential physical and
transition risks—and opportunities—across our
businesses. We are using the results to enhance our
strategy for climate change mitigation and adaptation,
and continue to integrate these considerations into
our business and investment approaches.
WE CONTINUED TO FOCUS ON INCREASING
DIVERSITY AT THE SENIOR LEVELS OF THE
ORGANIZATION
Having a diverse team where colleagues are engaged,
learning and developing reinforces our culture of
collaboration and strengthens our ability to build longterm value. We are focused on maintaining an
environment defined by deep relationships, strong
leadership, and disciplined talent attraction and
performance assessment processes. We continue to
strive for diverse representation within the firm’s
senior management, and we are making progress.
We have long been focused on increasing gender
diversity at Brookfield, and while our headcount has
more than doubled over the last five years, we have
grown our female representation at the most senior
levels of the organization by 170% during that time.
Specifically, female representation grew from 7% to
19% among managing partners and managing
directors. In 2021 alone, we increased female
representation in the managing partner/managing
director groups by 33%, largely through promotions.
Among all our employees globally, 46% are female.
We also continue to support ethnic diversity at all levels
of the firm, and in 2021 we launched a global process
for employees to self-identify their ethnicity. Almost
40% of employees working in our largest offices (100 or
more employees) self-identified as being part of an
underrepresented ethnic group, while approximately
18% of managing partners and managing directors in
those same offices self-identified as underrepresented.
Across all our offices globally, this percentage was 23%.
We will continue to monitor and report on our
advancements in this area.
LOOKING AHEAD
Acting responsibly is foundational to operating a
productive, profitable and sustainable business. We
have always believed that value creation and
sustainable development are complementary goals,
and our conviction has only grown stronger over the
course of our 100-plus years as an owner and operator
of long-term assets. We appreciate your help in
ushering in a more sustainable future for us all.
Bruce Flatt
Chief Executive Officer
9
INTRODUCTION
2021 ESG REPORT | Brookfield Asset Management
Who We Are
We are a premier global alternative asset manager with $720 billion of assets under
management across renewable power & transition, infrastructure, private equity, real
estate, and credit & insurance solutions. We are one of the world’s largest investors
in real estate, own the world’s largest private sector renewable power-generating
business, and hold a diverse array of infrastructure businesses and other privately
held companies that deliver essential products and services to our communities.
Given the nature of our portfolio, longevity and stability of operations are essential.
We are owner-operators in the way we manage
our business. We invest significant capital
alongside our investors, which creates a strong
alignment of interests. In addition, we have
developed strong operating capabilities over the
years, which are important for reinforcing
sustainable operations. Our goal is to enable
these businesses, as well as the communities in
which we operate, to thrive over the long term.
Our culture underpins everything we do. The
core principles underlying our approach include:
 Long-term perspective: Our long-term
approach influences everything we do,
including how we make investment decisions;
how we support and oversee our businesses
and measure their success; and how we
develop our people and compensate them.
 Alignment of interests: We ensure alignment of
interests with our investors in multiple ways:
1. We are compensated as an asset manager
through performance-based arrangements,
which are directly linked to increased value and
cash flows for our investors;
2. We invest significant balance sheet capital
alongside our investors;
3. Senior management has a significant economic
interest in the firm; and
4. Our employee compensation programs link a
significant portion of employee rewards to
successful investment outcomes.
 Collaboration: Our emphasis on fostering
collaboration enables us to benefit from a diverse
set of skills and experiences. Our talent
management processes and our approach to
long-term compensation encourage
collaboration—not just across our global asset
management business, but also with and among
our portfolio companies. This is demonstrated in a
number of ways, including in the sharing of
expertise and best practices through both formal
and informal channels, and by emphasizing
employee secondments and transfers as a means
of fostering employee development and
relationship building.
These principles inform how we manage the business
and are foundational to our success.
Management, officers and directors of the
Corporation and its affiliates hold direct,
indirect and economic interests representing
over 300 million Class A shares and share
equivalents of the Corporation.
2021 ESG REPORT | Brookfield Asset Management 10
ESG at
Brookfield
Our Guiding ESG Principles
ESG Affiliations and Partnerships
Stakeholder Engagement
About This Report
Comprehensive Topic Review and Analysis
11
ESG AT BROOKFIELD
2021 ESG REPORT | Brookfield Asset Management
Our Guiding ESG Principles
Our ESG strategy is centered on supporting business resilience and
creating value for our investors and stakeholders—now and in the future.
We manage our investments with integrity, combining economic goals with responsible citizenship. This is
consistent with our philosophy of conducting business with a long-term perspective in a sustainable and ethical
manner. It also requires operating with robust ESG principles and practices, and maintaining a disciplined focus
on embedding these into everything we do. Our approach to ESG is based on the following guiding principles:
Mitigate the impact of our
operations on the environment
 Strive to minimize the
environmental impact of our
operations and improve our
efficient use of resources over time.
 Support the goal of net-zero
greenhouse gas (GHG) emissions
by 2050 or sooner.
Ensure the well-being and safety
of employees
 Foster a positive work
environment based on respect for
human rights, valuing diversity
and having zero tolerance for
workplace discrimination, violence
or harassment.
 Operate with leading health and
safety practices to support the goal
of zero serious safety incidents.
Uphold strong governance practices
 Operate to the highest ethical
standards by conducting business
activities in accordance with our Code
of Business Conduct and Ethics.
 Maintain strong stakeholder
relationships through transparency
and active engagement.
Be good corporate citizens
 Ensure the interests, safety and
well-being of the communities in
which we operate are integrated
into our business decisions.
 Support philanthropy and
volunteerism by our employees. 
12
ESG AT BROOKFIELD
2021 ESG REPORT | Brookfield Asset Management
ESG Affiliations and Partnerships
Through our engagement with leading ESG frameworks and sustainability
organizations, we continue to be actively involved in discussions to advance ESG
awareness across private and public markets, and we are enhancing our ESG reporting
and protocols in line with evolving best practices. Below are some of the frameworks
and organizations with which we are affiliated.
In 2021, we joined the Net Zero Asset Managers (NZAM) initiative, which
consists of a group of international asset managers committed to
supporting the goal of net-zero greenhouse gas (GHG) emissions by
2050 or sooner, emphasizing our alignment with the Paris Agreement.
We have set an interim target to achieve an approximately two-thirds
reduction in Scope 1 and 2 emissions for $147 billion of AUM—
approximately one-third of our total portfolio—by 2030 or sooner.
We recently joined the Institutional Limited Partners Association (ILPA)
Diversity in Action (DIA) initiative, which brings together limited
partners and general partners who share a commitment to advancing
diversity and inclusion in the private equity industry. Joining the DIA
initiative underscores our commitment to advance diversity and
inclusion, both within our organization and the industry more broadly.
We have been signatories to the Principles for Responsible Investment
(PRI) since 2020, which reinforces our longstanding commitment to
responsible investment and ESG best practices. The PRI is one of the
world’s leading proponents of responsible investing, with an emphasis on
understanding the investment implications of ESG factors and supporting
an international network of investor signatories incorporating these
factors into their investment and ownership decisions.
In 2021, we became supporters of the Task Force on Climate-related
Financial Disclosures (TCFD). The TCFD aims to guide companies in
incorporating considerations relating to the effects of climate change
into business and financial decisions to help facilitate the transition
to a more sustainable, lower-carbon economy. We are aiming to
provide disclosures in respect of the 2022 fiscal year that adhere to
TCFD guidelines.
We are Alliance members of the Sustainability Accounting Standards
Board (SASB). SASB helps businesses around the world identify,
manage and report on the sustainability topics that matter most to
their investors. We utilize the SASB Engagement Guide as part of our
investment due diligence protocols.
NZAM
13
ESG AT BROOKFIELD
2021 ESG REPORT | Brookfield Asset Management
In addition to our own partnerships, many
businesses across our portfolios are associated with
industry organizations and frameworks that promote
responsible business practices. Examples include:
Many of our real estate portfolio
companies report voluntary
environmental disclosures through The
Global Real Estate Sustainability Benchmark (GRESB),
an investor-driven organization that assesses the
sustainability performance of real estate portfolios
and assets.
To help progress a commitment to protect
biodiversity, our renewable power portfolio
assesses nature-related physical and
transition risks and opportunities in line with the
Taskforce on Nature-related Financial Disclosures
(TNFD) recommendations. The TNFD is a risk
management and disclosure framework for
organizations to report and act on evolving naturerelated risks, with the ultimate aim of supporting a shift
in global financial flows away from nature-negative
outcomes and toward nature-positive outcomes.
Brookfield Properties, our real estate
operating company, has adopted the
WELL Health-Safety Rating, a global sustainability
designation that recognizes buildings that support
the long-term health and well-being of their
occupants. The rating was developed by the WELL
Building Institute, an organization focused on
improving human health and well-being through the
built environment.
Our reporting real estate entities
achieved an average global score of 85 in
2021, an increase of one point from 2020.
WELL Health-Safety Rating
To achieve the WELL Health-Safety Rating, buildings must demonstrate appropriate health and safety
measures, including cleaning and sanitization procedures, emergency preparedness programs, health service
resources, air and water quality management, and stakeholder engagement and communication.
Brookfield’s real estate businesses, health and safety standards align closely with the WELL Health-Safety
Rating. As outlined in our ESG principles, we seek to ensure the well-being and safety of our stakeholders by
operating with leading health and safety practices across our portfolio of assets. Creating a healthy and safe
environment at every property we own and operate is highly beneficial to our tenants, residents, communities
and employees of our business, who feel empowered and inspired by an environment that helps them achieve
their full potential each and every day. We believe the WELL Health-Safety Rating is a strong external and highly
regarded indicator, which validates our commitment to incorporating leading health and safety practices
across our portfolio.
The rating was developed during the emergence of the COVID-19 pandemic, with over 600 public health
experts incorporating components into the framework that evaluate a building’s implementation of strategies
protecting occupants from the spread of the COVID-19 virus.
WELL has specifically recognized our real estate operating company, Brookfield Properties, as an entity
progressing quickly in terms of the WELL Health-Safety Rating. In 2021, Brookfield’s real estate portfolio earned
a total of 190 WELL Health-Safety Ratings across multiple asset classes and regions around the world,
including the following portfolios:
BROOKFIELD REAL ESTATE
113
Retail Properties
58
U.S. Core Office
Properties
13
Canadian Core
Office Properties
5
Canary Wharf
(U.K.) Properties
1
United Arab Emirates
Core Office Property
14
ESG AT BROOKFIELD
2021 ESG REPORT | Brookfield Asset Management
Stakeholder Engagement
EMPLOYEES
 Employee Surveys
 Employee Engagement Groups
 Ethics Hotline/Whistleblowing Policy
 Internal Events/Town Halls
 Internal Communications/Intranet
INVESTORS
 Investor Meetings, Conferences, Webcasts and Calls
 Quarterly and Annual Reports
 Letters to Shareholders and Private Fund Investors
 Email Notifications and Updates
 Website/Private Fund Investor Portal
COMMUNITIES
 Community Engagement
 Philanthropy
We engage with our investors, employees and communities to ensure that our decisions are beneficial to the
interests of our business and to those of our stakeholders.
Through our comprehensive communications program, our stakeholders receive regular updates on our
performance and progress toward our goals. This includes meetings, webcasts, annual filings, press releases and
published reports such as our annual report and quarterly interim reports. This information and more can be
found on our website, as well as our investor portal. We strive for full transparency and make our management
available to communicate with investment analysts, financial advisors, rating agencies and the media.
Vodafone: Engagement With
Indigenous Communities
Brookfield Infrastructure Group's New Zealand
data distribution business, Vodafone, recently
launched a five-year strategy to improve
Indigenous cultural awareness, focusing on the
Māori community. This program aims to boost
Māori business and support digital inclusion and
a sustainable future for the community. To
support this effort, Vodafone has signed a
strategic relationship agreement for a mutually
beneficial relationship. In addition, Vodafone has
delivered cultural, language and leadership
programs, along with adding internal educational
resources for employees, and embedded the
use of Māori language in communications and
branding material, encouraging awareness
across all levels of the company.
BROOKFIELD INFRASTRUCTURE
15
ESG AT BROOKFIELD
2021 ESG REPORT | Brookfield Asset Management
About This Report
This report covers the Firm’s ESG-related activities
between January 1, 2021 and December 31, 2021,
unless noted otherwise.
In determining the most important issues to our stakeholders and the
industry, we referred to the Sustainability Accounting Standards Board
(SASB) standards for Asset Management and Custody Activities and the
Global Reporting Initiative (GRI), which we believe provide best-practice
guidance on ESG disclosures that are most meaningful for our business.
We continue to align our climate-related corporate reporting with the
recommendations of the TCFD and aim to provide formal TCFD disclosures
for the 2022 fiscal year in 2023.
We engaged with our stakeholders and referenced the aforementioned
reporting frameworks in identifying 17 topics that we consider to be
material to the resiliency of our business and our long-term success. These
topics are divided into four areas: Our Investment Approach, Building a
Better World, Our People, and Corporate Governance and Ethics.
16
ESG AT BROOKFIELD
2021 ESG REPORT | Brookfield Asset Management
Comprehensive Topic Review and Analysis
Based on guidance from SASB and the GRI, the following table sets out the topics that we believe are material to our business.
TOPIC DESCRIPTION
OUR INVESTMENT APPROACH
ESG Integration into Our Investment Process Incorporating ESG factors into investment decision-making and valuation, modeling, portfolio construction and engagement with
portfolio companies
Proxy Voting Guidelines and Stewardship
and Engagement Procedures
Managing our investment activities, including proxy voting, in the best interests of our investors, with the goal of protecting and
enhancing the long-term value of our investments
Systemic Risk Management Integrating financial and ESG-related due diligence, and risk and opportunity management into our overall risk management program
Sustainable Finance Offering our investors opportunities to contribute to a sustainable market economy and support adaptation to global climate change
challenges through sustainable investments
BUILDING A BETTER WORLD
Climate Change Strategy GHG Emissions: Working to reach net-zero emissions by 2050 or sooner by reducing our carbon emissions and accelerating the
transition to a net-zero carbon economy through our renewable power business and global transition investment strategy
Clean Energy: Accelerating the transition to a low-carbon economy through Brookfield’s renewable power operations
Green Building: Prioritizing the investment in, and development of, energy-efficient and responsible buildings and businesses
Water and Waste Measuring Brookfield’s water and waste usage, with the objective of conserving water and reducing waste
OUR PEOPLE
Human Capital Development Attracting people aligned with our culture and providing them with opportunities to develop to achieve high engagement, and strong
retention and facilitate smooth succession
Diversity and Inclusion Maintaining a work environment that benefits from different perspectives and reinforces our culture of long-term focus, aligned
interests and collaboration
Occupational Health and Safety Ensuring the health and safety of our workforce via both physical and mental health policies, goals and programs
17
ESG AT BROOKFIELD
2021 ESG REPORT | Brookfield Asset Management
TOPIC DESCRIPTION
CORPORATE GOVERNANCE AND ETHICS
Board Effectiveness Monitoring the effectiveness of Brookfield’s Boards through meetings, self-assessments, tracking attendance, training and
education, and other initiatives
Board Skills and Oversight Ensuring the Board possesses the relevant experience, expertise and skills to oversee and support Brookfield in its endeavors
Business Ethics Conducting Brookfield’s business ethically and responsibly within operations as well as business partnerships, including through the
Firm’s values, code of conduct, policies, and processes
Human Rights Ensuring that Brookfield conducts business in a manner that respects and supports the protection of human rights
Responsible Contracting Engaging contractors that conduct their business in a responsible manner as outlined in Brookfield’s responsible contractor policies
Audit Oversight Monitoring the effectiveness and compliance of the Board, management and portfolio companies
Executive Compensation Reinforcing long-term stewardship of the business through our approach to executive compensation
Data Privacy and Security Protecting the Firm, its assets and its data from data privacy threats through strong practices that are in compliance with evolving
regulatory requirements across our asset management operations
2021 ESG REPORT | Brookfield Asset Management 18
Our Investment
Approach
ESG Integration into Our Investment Process
Proxy Voting
Systemic Risk Management
Sustainable Finance
19
OUR INVESTMENT APPROACH
2021 ESG REPORT | Brookfield Asset Management
ESG Integration into Our
Investment Process
We integrate ESG into all aspects of investment decision-making and ongoing portfolio
management, including portfolio construction, financial models and business trends,
investment valuations, monitoring portfolio company performance and engaging with
their management teams.
DUE DILIGENCE
During the initial due diligence phase of an
investment, we proactively identify material ESG
risks and opportunities relevant to the potential
investment. In doing so, we leverage our
investment and operating expertise and utilize
industry-specific guidelines that incorporate SASB
guidance. In 2021, we enhanced our ESG due
diligence guidelines to more specifically address
climate change, human rights and modern slavery
risks. Where warranted, we perform deeper due
diligence, working with internal experts and
third-party consultants. With respect to climate
change, in addition to investing in green companies,
we also intend to invest significant capital in
opportunities to transition businesses from brown
to green or retire them in a responsible manner.
INVESTMENT COMMITTEE APPROVAL
All investments made by Brookfield must be
approved by the applicable Investment Committee,
which makes its decision based on a set of
predetermined criteria. To facilitate this process,
investment teams outline for the Committee the
merits of each transaction and material risks,
mitigants and significant opportunities for
improvement, including those related to ESG, such
as bribery and corruption risks, health and safety
risks, and environmental and social risks.
ONGOING MANAGEMENT
As part of each acquisition, investment teams
create a tailored integration plan that includes any
applicable material ESG-related matters for review
or execution. Brookfield looks to advance ESG
initiatives and improve ESG performance to drive
long-term value creation, as well as to manage any
associated risks. Our approach is informed by the
strong correlation—which we continue to witness—
between managing these considerations and
enhancing investment returns. It is the
responsibility of the management teams within
each portfolio company to manage ESG risks and
opportunities through an investment’s life cycle,
supported by the applicable investment team. The
combination of local accountability and expertise, in
tandem with access to Brookfield’s investment and
operating capabilities, is important when managing a
wide range of asset types across jurisdictions.
EXIT
When preparing an asset for divestiture, we create
robust business plans outlining potential value
creation deriving from several different factors,
including relevant ESG considerations. We also
prepare both qualitative and quantitative data that
summarize the ESG performance, where applicable, of
the investment and provide a holistic understanding of
how Brookfield has created value and managed the
investment during the holding period. 
20
OUR INVESTMENT APPROACH
2021 ESG REPORT
| Brookfield Asset Management
Proxy Voting
In early 2021, Brookfield established new Proxy Voting
Guidelines. These guidelines are intended to ensure
that we vote proxies in our investors’ best interests, in
accordance with any applicable proxy voting agreements
and consistent with the investment mandate.
Brookfield assesses a variety of ESG factors in determining whether voting a
proxy is in a client’s best interests, including, but not limited to, gender equality,
board of directors’ diversity, ecology and sustainability, climate change, ethics,
human rights, and data security and privacy. Our guidelines cover information
about our Proxy Voting Committee and conflicts of interest, as well as key voting
issues. These voting issues could include ESG issues, director elections, director
independence, board effectiveness and diversity, board compensation, overboarding and executive compensation, among other topics. The guidelines also
uphold our strong commitment to ESG practices, and our positions concerning
climate risk, human rights, and diversity and inclusion.
All proxy votes are reviewed and voted on by the leaders of the respective
business groups and their associated teams. These teams will retain ultimate
responsibility for determining how to vote each proxy, taking into consideration
the investment mandate, contractual obligations and a review of all relevant
information. In addition, the teams will track all proxies, document the basis for
the vote and serve as a record of Brookfield’s votes. As part of its Proxy Voting
Guidelines, Brookfield has created a Proxy Voting Committee to oversee proxy
voting across its holdings. Comprising senior executives across Brookfield, this
committee meets annually to review the guidelines, evaluate the effectiveness
of their implementation and confirm whether they continue to be designed to
ensure that proxies are voted in the best interests of investors.
21
OUR INVESTMENT APPROACH
2021 ESG REPORT | Brookfield Asset Management
Systemic Risk Management
Risk management is an integral part of our business and key to creating long-term
value for our investors.
We recognize that risks to our business—including
ESG-related risks—are constantly evolving, and our
program aims to monitor and proactively mitigate
and manage them over time.
As an asset manager, the objectives of our risk
management program are to align risk appetite and
business strategy, reduce operational surprises, allocate
resources effectively, enhance decision-making and
visibility, identify and manage risks efficiently, and
improve communication surrounding risk.
Our risk management program addresses strategic
and operational risks, with an emphasis on the
proactive management of both current and emerging
risks. We also monitor our risk program to address
the evolving needs of our business and ensure that
we have the necessary capacity to respond to
changes. In 2021, we continued to enhance our
portfolio-wide climate risk management methodology
and framework that identifies, assesses, monitors
and reports on physical and transition risks
associated with climate change. The framework
utilizes scenario analysis and defines a methodology
to ensure a comprehensive and comparable
inventory of risks.
A fundamental principle of our investment approach
is that risk should be managed as close to its source
as possible and by those who have the most
knowledge and expertise in the specific business or
risk area. Senior management and functional groups
in our portfolio companies are therefore responsible
for managing the risks facing their businesses and
tailoring a mitigation plan to each specific risk area.
Brookfield, in its capacity as an asset manager,
provides strategic input and support through
regular monitoring and reporting processes, and
facilitating appropriate coordination and sharing of
best practices, including through its representation
on boards of directors and other governance
structures. We regularly review our risk
management program and processes, including
those relating to ESG risks such as climate change,
and implement improvements, as required.
We have implemented strong governance practices
to monitor and oversee our risk management
program, including the management of ESG risks.
Brookfield’s Board of Directors oversees risk
management with a focus on more significant and
systemic risks and leverages management’s
monitoring processes, with oversight of specific risk
areas (including specific ESG risks, where applicable)
delegated to board committees. The Risk Management
Committee oversees management of significant
financial and non-financial risk exposures, reviews risk
assessment and risk management practices, and is
responsible for confirming that Brookfield has an
appropriate risk-taking philosophy and suitable risk
capacity. The Audit Committee oversees financial
reporting risks and associated audit processes. The
Management Resources and Compensation
Committee oversees risks related to succession
planning, executive compensation and other human
capital risks. The Governance and Nominating
Committee oversees risks related to governance
structure as well as Brookfield’s overall ESG strategy.
Brookfield provides regular updates on overall risks to
the Risk Management Committee, which includes
quarterly or semi-annual updates on Brookfield’s
current risk profile and emerging risks, including health
and safety, anti-bribery and corruption, disruption and
reputation, and periodic in-depth reports on specific
risk areas such as climate change.
22
OUR INVESTMENT APPROACH
2021 ESG REPORT | Brookfield Asset Management
Sustainable Finance
We continue to be an active leader in sustainable finance products, enabling
our investors and portfolio companies to contribute to a sustainable market
economy and support adaptation to global climate change challenges.
To support the global transition to sustainable
energy, Brookfield issues green bonds, preferred
shares and other instruments to fund the
development of green energy technologies and
to finance eligible investments. Brookfield has
established green bond frameworks and criteria
for green projects that align with the
International Capital Markets Association (ICMA)
Green Bond Principles. These include:
 Use of proceeds,
 Process for project evaluation and selection,
 Management of proceeds, and
 Reporting
For more information about our Green Bonds
Framework, please visit our 2021 Green Bond
and Preferred Securities report.
The evaluation and selection of green bonds and
other products is overseen by the Capital
Markets and Treasury (CMT) team, which includes
our Chief Financial Officer and senior executives.
In 2021, we issued approximately $8 billion in
green bonds, sustainability-linked debt and
green preferred securities, an increase from
$3.6 billion in the prior year.
Brookfield issued its first senior unsecured
green bond, a $500 million offering with a
10-year fixed-rate term. The transaction was
more than three times oversubscribed and
included more than 50 investors, which we
believe resulted in saving 5 bps on the
issuance. The proceeds of this green bond will
be allocated to the financing or refinancing of
recently completed and future eligible green
projects, such as green buildings, renewable
power, energy efficiency, and sustainable water
and waste management. In addition, Brookfield
Infrastructure Partners issued green preferred
units and Brookfield Renewable Partners
issued two green subordinated note offerings.
We continue to explore ways to expand our
portfolio of sustainable assets and investments
across renewable power & transition,
infrastructure, private equity and real estate.
ADDITIONAL INFORMATION
Brookfield's Green Bond Framework
In December 2021, Brookfield
Renewables closed a C$1.2 billion
project-level green bond for a 263 MW
hydroelectric portfolio in Quebec after
entering into a 40-year electricity
purchase agreement with Hydro Quebec.
In December 2021, our transition Fund
closed a $2.5 billion sustainabilitylinked subscription loan facility with 17
banks. The facility term is three years
and includes a 7.5 bps margin discount.
2021 ESG REPORT | Brookfield Asset Management 23
Building a
Better World
An Interview With Mark Carney and
Connor Teskey about the Transition to Net Zero
Climate Change Strategy
Water and Waste
24
BUILDING A BETTER WORLD
2021 ESG REPORT | Brookfield Asset Management
An Interview With Mark Carney and Connor
Teskey about the Transition to Net Zero
Mark and Connor discuss Brookfield’s long-term plan to reduce greenhouse gas emissions
Mark Carney
Vice Chair and Head of Transition Investing
Mark Carney is a Vice Chair of Brookfield Asset Management and Head of
Transition Investing. In this role, he is focused on the development of products
for investors that will combine positive social and environmental outcomes with
strong risk-adjusted returns. Mr. Carney is an economist and banker who
served as the Governor of the Bank of England from 2013 to 2020, and prior to
that as Governor of the Bank of Canada from 2008 until 2013. He was Chairman
of the Financial Stability Board from 2011 to 2018. Prior to his governorships,
Mr. Carney worked at Goldman Sachs as well as the Canadian Department of
Finance. He is a long-time and well-known advocate for sustainability, specifically
with regard to the management and reduction of climate risks, and is currently
the United Nations Special Envoy for Climate Action and Finance.
Connor Teskey
Managing Partner, CEO Renewable Power & Transition
Connor Teskey is a Managing Partner, Head of Brookfield’s Renewable Power
& Transition Group, and Chief Executive Officer of Brookfield Renewable
Partners. Mr. Teskey is also Head of Europe for Brookfield Asset Management,
responsible for corporate operations and oversight across Brookfield’s
business in the region. Prior to these roles, Mr. Teskey was Chief Investment
Officer of the Renewable Power business. He also held roles focused on
investments, financing and restructuring for both Brookfield’s private equity
funds and Brookfield Asset Management.
Connor: Mark, we were really excited to have you join
us in 2020. Like many, we applauded your efforts to put
climate change closer to financial decision-making, but
what made you interested in joining Brookfield?
Mark: I began my career in finance before switching
to the public sector and I’ve always believed in the
power of bringing those two areas closer together.
Brookfield is one of the world’s largest asset managers
and I liked the ethos of long-term investing in the real
assets that form the backbone of our economies.
Connor: You and I have had lots of questions about
what transition investing really is. It’s a relatively new
area, so can you say a few words about what that
means?
Mark: This is all about the how the world moves
from the carbon-intensive economies of today to a
world of net-zero emissions by the middle of the
century. The physics of climate change are stark and
tell us that we have a finite carbon budget left to
spend before the world starts potentially tipping into
dangerous global warming. We’ve got to take action
where the emissions are highest and substantially
reduce them over the next 30 years. 
25
BUILDING A BETTER WORLD
2021 ESG REPORT | Brookfield Asset Management
Connor: This isn't the traditional territory of a central banker. What led you into
this space?
Mark: Reducing emissions as quickly as possible will protect our environment
but crucially, it also protects our financial assets. A world of rapid warming will
cause new financial burdens, create strain on public services and risks leaving
trillions of dollars of capital investment stranded. It struck me hardest when, as
Governor of the Bank of England, with responsibility for regulating the world’s
fourth largest insurance industry, we began to see that future climate impacts
were beginning to show up as a growing threat to their underwriting
assumptions and hence their financial stability. Among other things, that
convinced me that we cannot treat climate change and the financial markets
independently anymore.
Connor: You and I work very closely on the Brookfield Global Transition Fund, but
we are also aware of the broader Firm’s efforts—and part of that includes joining
the Net Zero Asset Managers (NZAM) initiative. Can you talk a bit about that?
Mark: As you know, Brookfield aims to reach net-zero greenhouse gas
emissions across the business by 2050, or sooner. We have solidified this
commitment by joining the NZAM which requires members to set meaningful
interim targets to drive change across a defined proportion of their assets. We
have set our initial target to reduce emissions by approximately two-thirds by
2030 across roughly one-third of our total assets, amounting to $147 billion.
Connor: That’s a substantial target. Why did we pick that number?
Mark: We wanted it to be meaningful but also recognize that transition won’t
happen overnight. We need to pick the areas where we can have nearer-term
impact because the technologies are mature or the investments yield value more
quickly. That means focusing on the assets where Brookfield has control—which
is around 70% of our assets under management—and then go through their
business plans meticulously to prepare them for the necessary investments.
Connor: What kind of investments will that require?
Mark: It’s going to require a range of actions, including operational efficiencies,
greener production processes and switching from fossil fuels to renewable
power. Of course, Brookfield knows a lot about delivering renewable power.
Connor: Absolutely. We have a unique advantage at Brookfield of being an investor in
the renewable energy sector for many decades. Our portfolio of 21 GW makes us one
of the largest pure-play renewable power companies in the world and we’re
committed to doubling our fleet capacity by 2030. We’ve become one of the biggest
contractors of renewable power to companies globally, with more than 700 industrial
and commercial clients.
Mark: And buying renewable power is usually the first major step a business can
take to significantly reduce its emissions.
Connor: That’s right. So, once you’ve developed a meaningful relationship and
credibility with these companies, there are opportunities to support them reduce their
emissions even further, through things like hydrogen, carbon capture and investing in
greener industrial processes like electrification.
Mark: That next opportunity is one of the reasons I’m so excited by the Brookfield
Global Transition Fund, which has raised a record $15 billion to invest in a wide
spectrum of decarbonization technologies and partnerships. As I said, transition
doesn’t mean flipping a green switch or investing only in companies that are already
green. Financial institutions must go where the emissions are and back companies—
including heavy-emitting sectors like steel, cement, and transportation—that have
credible plans to transform their business for a net-zero world.
Connor: We’re seeing a lot of interest in the Fund, and our ability to support
companies in those hard to abate sectors is creating a lot of interest in many different
sectors around the world. When we think about measuring impact, it will mean
thinking a bit differently about how we show that journey from a carbon-intensive to
lower-carbon business.
Mark: Which is why the reporting and disclosure is so important for investors. I
spent many years leading the Taskforce for Climate-related Financial Disclosures
(TCFD) precisely because it provides that link between financial investors and the
impact of climate change on their portfolios. Brookfield is preparing for its first TCFD
Report in 2023, which, alongside the annual ESG Report, will give our investors the
clear picture they need to make smart decisions.
Connor: It’s going to be a great year ahead. Thank you, Mark.
26
BUILDING A BETTER WORLD
2021 ESG REPORT | Brookfield Asset Management
Climate Change Strategy
We believe that there is a global transition to a net-zero economy underway, which is estimated to require an investment
of $3.5–5.0+ trillion annually. We believe we are well positioned to support this initiative and are taking a proactive
approach to transition our portfolio of investments accordingly. Climate change significantly impacts both our business and
the communities in which we operate around the world. As part of our climate change mitigation strategy and our efforts
to build business resilience, we support a goal to achieve net-zero greenhouse gas (GHG) emissions by 2050 or sooner.
To ensure that our portfolio aligns with climate action
best practices, we made a commitment to support the
goal of reaching net-zero emissions by 2050 or sooner
across all assets under management; created the
Brookfield Global Transition Fund to source
opportunities underpinned by a decarbonization
objective and deliver solutions that facilitate the
transition to net-zero; and continue to align our
practices with the TCFD.
BROOKFIELD’S NET-ZERO COMMITMENT
Brookfield has spent over 25 years building one of
the largest private renewable power businesses in
the world. We continue to increase the scale of our
renewable power operations and capacity to further
the goal of reducing dependence on fossil fuels and
the rapid reduction of over 70% of global emissions
from final energy consumption. Our renewable
energy businesses have a portfolio of 21GW of
operating renewable assets, more than enough to
support the demand of the city of London, and we
aim to double that capacity by 2030.
Our 2030 net-zero interim target currently
encompasses $147 billion of our assets under
management[5]. We have set our interim target to
reduce Scope 1 and 2 emissions of these in-scope
assets by approximately two-thirds (approximately
one million metric tons of CO2
e) across our
renewable power & transition, infrastructure,
private equity and real estate businesses, focusing
our initial efforts on those investments where we
can exercise significant influence over the strategy
and execution. Work is well underway in our
development of a comprehensive inventory of
emissions across our businesses from which we can
measure and report emissions and develop specific
decarbonization plans and related targets as
appropriate. Our objective is to substantially
increase the proportion of our in-scope assets over
time, consistent with our ambition to achieve
net-zero across all our assets under management
by 2050 or sooner. Our emissions inventory will
include material Scope 3 emissions at a future time
when we are able to gather complete and
sufficiently high quality data.
Our strategy is to leverage our position as a global
leader in renewable power to accelerate the
transition to net zero.
Net Zero Asset Managers Initiative
In addition to continuing to make major investments
in renewable energy globally, we will manage our
investments to be consistent with the transition to a
net-zero economy.
In 2021, we formalized our commitment to net-zero
by becoming a signatory to the Net Zero Asset
Managers (NZAM) initiative. Our objectives in joining
NZAM include:
 Driving progress on decarbonization goals
consistent with an ambition to reach net-zero
emissions by 2050 or sooner across all assets
under management;
 Setting an interim target of a specific proportion of
our assets to be managed in line with net-zero, with
targeted emissions reduction by 2030; and
 Reviewing this interim target at least every five
years, with a view to increasing the proportion of
AUM covered until 100% of assets are included.
5 AUM figures are as of Q4 2020 to align with the 2020 baseline year
emissions used for our interim target.
27
BUILDING A BETTER WORLD
2021 ESG REPORT | Brookfield Asset Management
We Will Help Accelerate the Transition to Net Zero
We will catalyze companies onto net-zero pathways
aligned with the Paris Agreement through our new
global transition investment strategy, focusing
specifically on investments that will accelerate the
transition to a net-zero carbon economy.
We Will Collaborate
We will work with leading private-sector initiatives to
advance the role of finance in supporting the
economy-wide transition, to accelerate capital flows
consistent with the Paris Agreement and to promote
widespread adoption of decision-useful
methodologies to support credible transition
planning, analysis and investing.
We Are Committed to Transparency
 We will track and report GHG emissions consistent
with GHG Protocol and PCAF standards;
 We will publish decarbonization plans every five
years consistent with the Paris Agreement; and
 We continue to align our business with the TCFD
recommendations and are targeting to incorporate
TCFD disclosures for the 2022 fiscal year.
We Will Continue to Pursue
Industry-Leading Returns
We will continue to pursue industry-leading returns
for our investors, consistent with our long track
record of building the backbone of a more
sustainable global economy.
BROOKFIELD GLOBAL TRANSITION FUND
We have launched our $15 billion Brookfield Global
Transition Fund (BGTF), the first in a series of funds
for our transition strategy that is dedicated to
investments supporting the global transition to a
net-zero economy. BGTF will build on Brookfield’s
leadership in renewable power and deep
operational capabilities to scale clean energy and
drive the transformation of carbon-intensive
businesses to achieve alignment with the Paris
Agreement. BGTF investments are expected to
contribute to select UN Sustainable Development
Goals (SDGs), which are the most globally recognized
benchmark in the impact and ESG space. Consistent
with its dual objectives of earning strong riskadjusted returns and generating a measurable
positive environmental change, the Fund will report
to investors on both its financial and environmental
impact performance. We will follow leading global
standards for impact measurement and management
and disclose quantitative impact results.
The BGTF investment strategy centers on three
primary areas:
1. Business Transformation: Transition utility,
energy and industrial businesses driving carbon
dioxide equivalent reduction and decreased
energy consumption through investment in
greener production processes and energy
efficiency. This will include investments in
non-pure-play renewable opportunities, utility
companies, industrial and energy-efficiency
technologies such as smart meters and electric
vehicle charging stations.
2. Clean Energy: Aid in the development and
accessibility of renewable energy sources. This
will include hydroelectricity, wind and solar
development, green hydrogen, battery storage,
electrical grid and distribution, and smart grids.
3. Sustainable Solutions: Solutions that drive the
growth of a circular economy in areas such as
waste management, resource efficiency and the
development of resilient infrastructure. This will
include heating and cooling, clean water
concessions, waste management technology,
recycling and waste/sewage utilities.
28
BUILDING A BETTER WORLD
2021 ESG REPORT | Brookfield Asset Management
TASK FORCE ON CLIMATE-RELATED
FINANCIAL DISCLOSURES (TCFD)
In 2021, we continued our work to align with the
TCFD recommendations and are targeting to publish
TCFD disclosures for the 2022 fiscal year. We had
previously assessed our practices against the
recommendations and developed an implementation
roadmap for alignment. During the past year, we
continued to make progress on our implementation
roadmap related to climate strategy, risk
management, and metrics and targets, including:
 Committing to the Net Zero Asset Managers
initiative and beginning work on targets and plans
for aligning to the net-zero goal
 Successfully completing fundraising and
commencing investment activities for the
Brookfield Global Transition Fund, which has a
mandate to focus on investments that contribute
to the transition to a net-zero global economy
 Enhancing and further operationalizing our climate
risk management methodology and framework
 Undertaking a comprehensive Phase 1 climate risk
assessment to better understand the potential
physical and transition risks, as well as
opportunities, across our businesses
 Leveraging the Phase 1 climate risk assessment
results to identify ways to improve our approach to
climate risk management and continuing to
integrate these considerations into our business
and investment strategies
 Formally incorporating climate risk and
opportunity considerations into our ESG Due
Diligence Guidelines, and adopting tools to assist
with the assessment
 Implementing an approach for ongoing GHG
reporting across our portfolio consistent with the
GHG Protocol
 Expanding the use of metrics and targets, including
those related to GHG emissions, water and waste
CLEAN ENERGY
As the global economy transitions away from a
reliance on fossil-fuels and moves toward a lowcarbon economy, our renewable power and clean
energy generation business continues to grow and
increasingly support the decarbonization of global
electricity grids.
In 2021, Brookfield Renewable’s power generation
helped to avoid approximately 29 million metric tons[6]
of carbon dioxide net emissions equivalent (mtCO2
e).
6 Our avoided emissions are based on our long-term average generation and
the Global Grid Average Emission Factor (IEA 2021)
Brookfield Renewable aims to develop
an additional 21,000 MW of new clean
energy capacity by 2030, doubling the
portfolio to 42,000 MW.
Arc: Investing in Renewable Energy
Brookfield Infrastructure’s Australian rail
business, Arc Infrastructure, has replaced
approximately 32 km of overhead power lines
with a solar powered system. This will assist in
both reducing greenhouse gas emissions and
increasing the reliability of the electric grid in
times of severe weather or bush fires. To further
lower the impact of this project on the
environment, the displaced materials from
overhead power lines were melted down and
recycled. This project represents one of many
examples of how Brookfield encourages its
companies to integrate tangible environmentally
beneficial initiatives into their growth and
maintenance capital programs, where feasible.
BROOKFIELD INFRASTRUCTURE
29
BUILDING A BETTER WORLD
2021 ESG REPORT | Brookfield Asset Management
Isagen: Climate Change Analysis
Since 2015, our regional business in Colombia has partnered with the National
University of Colombia (“UNAL”) to conduct water resource variability analysis
on 3,000 MW of hydroelectric assets at five facilities across the country. Over
the past three years, efforts have been focused on conducting forward-looking
climate change analysis.
Together, we have built a model based on scientific data from global climate
models, coupled with local regional models, including historical land cover
changes being projected to 2040 and 2090. Downscaling of climate
scenarios has been applied to hydrological models to project stream flows.
While our work is ongoing, the findings to date have noted no direct impacts
due to changes in climate variables—temperature, humidity or wind speed.
There were potential changes projected in hydrological variables—
precipitation, stream flow and sediment transport. Examples include average
changes of 10–30% of annual precipitation and an intensification of rainfall
events in certain river systems. The initial results also indicated an increase
in sediment production from extreme precipitation and changes in land use.
More work will be conducted as certain local conditions are still not well
represented in climate models.
In addition, we assessed the five facilities’ ability to withstand the potential
projected hydrological variables from 2040 onwards and determined that
the assets are well-positioned to manage these risks given their design
standards together with their ongoing capital expenditure program.
BROOKFIELD RENEWABLE
30
BUILDING A BETTER WORLD
2021 ESG REPORT | Brookfield Asset Management
Water and Waste
Water and waste reduction are an integral part of our efforts to minimize our
environmental impact across Brookfield and its portfolio companies. We employ
best practices to optimize, evaluate and continuously improve our approach to
the use of shared resources and management of waste.
Our green building initiatives cover topics including energy reduction, water conservation, recycling,
enhanced indoor air quality, alternative transportation parking, environmentally friendly cleaning materials
and erosion control.
A Commitment to Green Buildings
190
certifications in the U.S.
and Canada
100
certifications[7]
51
certifications in the U.S.
60
office certifications in the
U.S. and Canada
7 LEED® is the preeminent program for the design, construction, maintenance and operations of high-performance green buildings. LEED®, and its related logo, is a
trademark owned by the U.S. Green Building Council and is used with permission—usgbc.org/LEED
CERTIFIED
SUSTAINABLE
PROPERTY IREM®
56
certifications in our real
estate portfolio
6
certifications
36
certifications
32
certifications
Note: Reflects certifications across Brookfield’s entire real estate portfolio.
Schoeller Allibert: Business
Model and Innovation
Schoeller’s mission is to design and produce
packaging products that respect the
environment by reducing environmental stress
caused by packaging waste. Schoeller has been
inventing, developing, designing and
manufacturing Returnable Transit Packaging
(RTP) for more than 60 years. The company’s
products are 100% recyclable and designed to
be durable and support multi-use and
optimized lifespans, which can be up to 10
years in industrial conditions. Studies show that
the carbon footprint of reusable crates is
between 60 to 80% lower than single-use
alternatives, such as cardboard boxes.
The company was one of the first RTP
producers to be accredited by the European
Food Safety Authority (EFSA) for recycling
food-grade high-density polypropylene (HDPE)
and polypropylene (PP) crates into new
containers for food contact. In addition, the
company offers an optional rental program that
enables customers to contribute to the circular
economy and adopt a low-carbon approach.
BROOKFIELD PRIVATE EQUITY
31
BUILDING A BETTER WORLD
2021 ESG REPORT | Brookfield Asset Management
Michipicoten Hydroelectric Station:
Rescuing Fish in Canada
Every fall, salmon and rainbow trout spawn in the area downstream of our
Michipicoten hydroelectric station in the province of Ontario. However, in
2021, due to extreme drought, this spawning area became isolated from the
main river channel. To mitigate impacts on the stranded fish, we collaborated
with the Ministry of Northern Development Mines and Natural Resources to
relocate fish to the main river channel. Due to our efforts, we relocated almost
2,000 fish representing 12 different species.
BROOKFIELD RENEWABLE
2021 ESG REPORT | Brookfield Asset Management 32
Our People
Human Capital Development
Diversity, Equity and Inclusion
Occupational Health and Safety
33
OUR PEOPLE
2021 ESG REPORT
| Brookfield Asset Management
Human Capital
Development
We value our people and support their long-term
success by seeking opportunities for them to grow and
develop professionally. This reinforces strong succession
and ensures that we maintain an engaged workforce.
Our employees drive our success and ensure that we deliver on our
commitments to investors and other stakeholders. We seek to create a
positive, open and inclusive work environment that enables employees
to develop. Inclusive leadership and disciplined talent management
processes are critical to our success in this regard.
Inclusive leadership starts with a strong tone at the top. Our Code of
Business Conduct and Ethics and Positive Work Environment Policy set a
consistently high standard for how we interact with one another and
reinforce a work environment conducive to learning and development.
To accomplish this, we focus on developing our people leaders:  Ensure the mandate of a people leader is clear: to provide a work
environment that is conducive to learning and development and in
which people feel safe when stepping outside their comfort zone. This
is critical to our success in developing our people.
 Provide training that clearly outlines the key elements of an
environment that supports development.
 Provide feedback to our people leaders to enhance their development. 
34
OUR PEOPLE
2021 ESG REPORT | Brookfield Asset Management
Disciplined talent management processes also provide support to our
people leaders and enhance our success in developing our people:
RECRUITMENT
We proactively recruit people who are aligned with our culture and
have the potential to grow and develop within the Brookfield
organization. This includes ensuring diverse representation. Following
are key activities that have been instrumental in our progress:
 Taking the time required to ensure a diverse slate of candidates by
gender. We have expanded our focus to include ethnic diversity and
have begun tracking the ethnicity of candidates.
 Developing objective criteria for each role to evaluate all candidates.
 Ensuring female representation within the Brookfield teams that
interview candidates and ultimately make the hiring decision.
50%
of our hires in 2021 were female, including 40% in
investment/finance and 60% in all other functions
>75%
of the positions hired in 2021 included at least two
female candidates, an increase from 70% in 2020
PERFORMANCE MANAGEMENT
We continue to add discipline to our process for assessing performance
and potential.
 In 2021, we finalized the performance criteria for virtually all roles.
These criteria clearly define what good performance entails and enable
objective and consistent assessments across Brookfield. They also clarify
the key indicators required for promotion to the next level.
 Annually, we provide training for people leaders on how to assess their
team members, mitigate the impact of bias in their assessments and
provide constructive feedback that is clear and focused on development.
41%
of all promotions in 2021 were female, including 33%
in investment/finance and 63% in all other functions
>30%
of our managing partner/managing director promotions were female,
increasing our female representation at this level by approximately 33%
We continue to benefit from strong retention. Our managing partners
have worked together for 13 years on average and our senior leadership
team has more than 18 years of experience working together. 
35
OUR PEOPLE
2021 ESG REPORT | Brookfield Asset Management
The combination of recruiting the right people and the
discipline in our performance assessment process are
key factors in our ability to develop our people and
retain strong performers. Our grow-from-within talent
strategy prioritizes internal mobility to provide
opportunities to expand professional experience and
enhance collaboration across the business. This includes
transfers between geographies, business groups,
functions and to or from portfolio companies. Over the
last five years we have more than doubled our employee
population, which means we have many people in new
roles. An additional 16% of our population has taken on
new opportunities under our internal mobility program
and over 40% of those opportunities were provided to
females in 2021. We launched a number of new
businesses, including the insurance solutions and
transition businesses. These new businesses have led to
a large number of opportunities to transfer people
between business groups.
Breakdown of 2021 Internal
Mobility Opportunities Provided
Geographic Relocations: 20%
Between Functions: 4%
To or From Portfolio
Companies: 7%
Between Business Groups: 69%
ADDITIONAL INFORMATION
Employee Engagement Groups
Positive Work Environment Policy
Code of Conduct
Partner to Empower Program
In 2021, Brookfield's Real Estate Group launched Partner to Empower; a program designed to support Black
and minority business owners who want to open and maintain successful brick-and-mortar stores in our retail
locations. Our goal is to ensure that our retail locations are reflective of the diversity of the communities in
which they operate.
The program provides two types of support: 1) financial support, and 2) resources to provide guidance and
build strong networks to support success as follows:
 Brookfield has committed $25 million to this program over the next five years, which will be distributed to
support the buildout and merchandising of physical retail stores as they open in our shopping centers.
 Brookfield has partnered with 32 professional and financial services firms and educational institutions
to launch a variety of workshops to support opening a business. These workshops cover a range of
topics including:
» Developing business models
» Business law, leases and risk management
» Attracting investment
» Leadership and talent management
» Market selection and expansion strategies
» Store design and merchandising
» Business banking
» Franchising
» Human resources and staffing
» Marketing, branding and public relations
» Sales systems and inventory management
» Store operations
In the first year of operation, 40 businesses within 25 communities in the Southeastern United States
participated in this program. Sixty percent of these businesses are owned by females and 25 have opened
or are well on their way to opening their stores. In 2022, Brookfield expects to expand the program to the
Northeast and Midwest, and open 50–75 new stores.
Brookfield’s long-term goal is to expand the program nationally by 2025. We expect this program to support
250 new store openings and create in excess of 750 new jobs. Our success in executing this program will go
beyond the opened businesses, by creating an ecosystem of support between the current business owners
and our partners, the local community and members of the Black and minority communities.
BROOKFIELD REAL ESTATE
36
OUR PEOPLE
2021 ESG REPORT | Brookfield Asset Management
Diversity, Equity and Inclusion
A focus on diversity and inclusion reinforces our culture of collaboration and
strengthens our ability to develop our people, maintain an engaged workforce
and create value for our investors.
Our approach to diversity and inclusion has been deliberate and is integrated into our human capital
development processes and initiatives.
Over the past five years, our primary focus has been on gender diversity. Our efforts have led to a significant
increase in female representation at the senior levels. We have more than doubled our employee population
and have significantly increased our female representation at the most senior level of the organization during
this period: managing partner/managing director female representation increased from 7% to 19%. In addition,
senior vice president representation increased from 15% to 33% during this time. The discipline embedded
into our recruiting and performance management processes has been instrumental in this progress.
AT BROOKFIELD, WOMEN COMPRISE
60%
of our independent board
directors and 38% of all
board directors
46%
of our overall workforce
27%
of managing partners,
managing directors and
senior vice presidents
AT OUR PORTFOLIO COMPANIES, WOMEN COMPRISE
19%
of CEOs/heads of businesses
28%
of senior leadership
37
OUR PEOPLE
2021 ESG REPORT | Brookfield Asset Management
In addition to the aforementioned human capital development activities, we
undertake other activities to reinforce the importance of diversity and inclusion
in our business.
The following are a few examples:
 In March 2021, we launched a global process for employees to self-identify
their ethnicity. This information will assist us in identifying specific areas of
focus related to increasing ethnic diversity. Our response rate in the
countries where we have more than 100 employees (U.S., Canada, Australia,
the U.K. and Brazil) was 92%. Our results demonstrate our diversity.
 We support a number of Employee Resource Groups that are organized by
employees around shared interests, characteristics or experiences. We
established a structure for each of these groups to ensure the mandate is
clear, aligned with our values, appropriately supported by the organization,
and provides opportunities to demonstrate leadership, develop
relationships and collaborate.
Global Ethnic Diversity Metrics[8]
White: 52%
Asian: 28%
Black: 4%
Hispanic/Latinx: 3%
Two or More Races/Other: 7%
Did Not Respond or Declined to Self-Identify: 6%
8 As of March 31, 2022
IN COUNTRIES WHERE WE HAVE OVER 100 EMPLOYEES
(U.S., CANADA, AUSTRALIA, THE U.K. AND BRAZIL),
UNDERREPRESENTED ETHNICITIES REPRESENT:
39%
of our employee
population
18%
of managing partners/
managing directors
29%
of the investment team
44%
of the operations team
38
OUR PEOPLE
2021 ESG REPORT | Brookfield Asset Management
EMPLOYEE ENGAGEMENT GROUPS
Brookfield Women’s Network provides learning and networking opportunities for
women in various roles and at all levels of the company, across Brookfield’s
business groups.
The Brookfield Pride Network is focused on fostering a culture of inclusion for
LGBTQ+ employees and allies, providing support and a sense of community while
empowering employees to bring their whole selves to work.
The Brookfield Black Professionals Network focuses on attracting and retaining
Black professionals and aims to enhance the awareness and inclusiveness of our
workforce, while providing a forum for employees to share and learn from the
experience of others.
The Asian Professionals Network serves as an employee engagement group for
employees and allies of the Asian American and Pacific Islander community.
Brookfield Cares is the corporate philanthropic program for Brookfield employees.
Philanthropic activities are an important aspect of employee engagement; they
enable our employees to build meaningful relationships, foster personal growth
and they benefit the communities in which we operate. Our global matching
program allows employees to donate to a not-for-profit of their choice and
Brookfield will match their donation. In addition, each office has a dedicated capital
pool to support the causes that are most important to our people.
Brookfield Next Generation (bNEXT) brings together colleagues in the early stages
of their careers who want to engage with and learn from each other.
Everise: Employee Engagement,
Diversity, Equity and Inclusion
Everise continues to stress the importance of a
diverse workplace by ensuring inclusivity within
all levels of the company. Currently, over 62
nationalities are employed, with the capability
to provide support in 32 languages. Examples
of the company’s workforce diversity are:
 62% of the workforce comprises women, with
40% in leadership positions
 5% of the workforce comprises mature-aged
employees
 1.5% of the workforce consists of people
with disabilities
Everise has been recognized with the Excellence
in Diversity & Inclusion award from HR Excellence
Awards APAC, Achievement in Diversity & Inclusion
from the Stevie Awards for Great Employers and
Best Company to Work For award from HR Asia.
BROOKFIELD PRIVATE EQUITY
39
OUR PEOPLE
2021 ESG REPORT
| Brookfield Asset Management
Occupational
Health and Safety
Our goal is to have zero serious safety incidents by working
toward implementing consistent health and safety principles
across the organization.
Our health and safety policies and procedures apply not only to employees, but
also to contractors and subcontractors.
Our portfolio companies’ senior leadership teams are responsible for each
company’s health and safety performance, and their boards of directors oversee
their health, safety and security risk management efforts. Our portfolio company
CEOs provide quarterly reports to their respective boards of directors on:  Safety performance and incidents;  Results from internal or external program assessments; and  The status of improvement initiatives.
Our Safety Steering Committee sponsors our health and safety governance initiatives,
which aim to build a strong health and safety culture, encourage the sharing of best
practices, support the continuous improvement of safety performance and help
eliminate serious safety incidents throughout our portfolio of businesses. The
committee includes members of our senior leadership team and provides regular
progress reports to Brookfield’s Board on our health and safety initiatives.
~1.4M
hours of occupational health and safety training
completed across Brookfield’s portfolio companies
40
OUR PEOPLE
2021 ESG REPORT | Brookfield Asset Management
HEALTH AND SAFETY FRAMEWORK
Our health and safety framework is based on the
following principles:
 Senior executives are accountable for the health
and safety of their individual businesses
 Systems are tailored to company-specific risks and
integrated into the management of the business
 Performance is measured and systems are reviewed
regularly to identify areas for improvement
 Policies and procedures apply to employees,
contractors and subcontractors, and take into
consideration the protection of the public in general
 Training programs ensure that employees have
the necessary skills to conduct their work safely
and efficiently
 If a serious safety incident occurs, senior leadership
of the individual business conducts an in-depth
investigation to determine root causes and
formulate remediation actions
 Transparency and learning from experience
are promoted to continuously improve systems
and performance
Saeta: Reducing Health and Safety Risk in Spain
Vegetation control is an important consideration for utility-scale solar plant operation and maintenance.
Unchecked vegetation growth can lead to shading of the solar panels, which decreases the plant’s
productivity and can hinder our technician’s ability to work safely. In common practice, vegetation
management programs are usually performed with techniques that have associated risks in terms of health,
safety and environment. To address these risks, Saeta’s operations have implemented an alternative
vegetation management program by allowing sheep to graze at the solar plant sites. The program has
multiple direct and indirect benefits to our operations and local communities in proximity to the assets:
 Leads to fewer emissions due to reduction in chemical and machinery use
 Reduces wildfire risk and maintains the quality of the soil around the assets
 Supports the livelihood of local shepherds by providing a permanent place to hold sheep and reducing
feeding costs by up to 70%
BROOKFIELD RENEWABLE
41
OUR PEOPLE
2021 ESG REPORT | Brookfield Asset Management
Governance
Corporate Governance and Ethics
Brookfield Asset Management Board of Directors
ESG Governance
Business Ethics
Human Rights and Modern Slavery
Responsible Contracting
Audit Oversight
Executive Compensation
Data Privacy and Security
42
GOVERNANCE
2021 ESG REPORT | Brookfield Asset Management
Corporate Governance and Ethics
We recognize that strong governance is essential to sustainable business
operations, and we aim to conduct our business according to the highest ethical
and legal standards.
Our governance practices inform the way we conduct
our business and are designed to align with the
priorities of our investors. We continue to adapt and
enhance our policies to meet evolving standards and
regulations in our industry, including legislation,
guidelines and practices in all jurisdictions in which
we operate.
In 2021, 100% of our portfolio companies had
an Anti-Bribery and Corruption Policy and a
Code of Conduct.
Recent regulatory developments included the E.U.
Sustainable Finance Disclosure Regulation, E.U.
Taxonomy Regulation and U.K. TCFD and
Taxonomy, as well as the newly announced
International Sustainability Standards Board (ISSB).
We seek to continuously improve and refine our
processes by actively participating in the
development and implementation of new industry
standards and best practices.
Our corporate governance policies and practices are
comprehensive and consistent with the guidelines for
improved corporate governance in Canada adopted
by the Canadian Securities Administrators and the
Toronto Stock Exchange, as well as the requirements
of the U.S. Securities and Exchange Commission, the
New York Stock Exchange, and the applicable
provisions under the U.S. Sarbanes-Oxley Act of 2002.
We continuously assess our governance practices and
disclosures with specific attention to evolving
Canadian and U.S. guidelines, as well as developments
in other jurisdictions in which we operate.
Brookfield is committed to conducting its business
activities with honesty and integrity, and in compliance
with applicable legal and regulatory requirements.
During the course of 2021, we enhanced our vendor
management program, including developing a Vendor
Code of Conduct that sets out our expectations of
vendors that provide goods or services to Brookfield,
including, where applicable, to have the necessary
policies and procedures in place to support such
commitments within their supply chain.
In connection with any vendor engagement, we must
comply with all policies and procedures, including the
following to the extent applicable:
 Anti-bribery and corruption program
 Data protection program
 Enterprise information security policy
 Anti-money laundering and trade sanctions policy
 Anti-slavery and human trafficking policy
ESG POLICY
In 2022, we developed a global ESG Policy that
formalizes our practices related to operationalizing
our ESG principles. This document codifies our
longstanding commitment to integrating ESG
considerations into our decision-making and day-today asset management activities. At Brookfield,
sound ESG practices are integral to building resilient
businesses and creating long-term value for our
investors and other stakeholders.
Certain of our publicly traded controlled affiliates
maintain ESG policies aligned with the provisions of
the ESG Policy but reflecting factors applicable to
their respective investment strategies. As discussed
previously, we have also continued to strengthen our
ESG governance by enhancing our firm-wide ESG
Due Diligence Guidelines.
ADDITIONAL INFORMATION
Statement of Corporate Governance Practices
Additional Governance Documents
Vendor Code of Conduct
ESG Policy
43
GOVERNANCE
2021 ESG REPORT
| Brookfield Asset Management
Brookfield Asset
Management Board
of Directors
Our Board of Directors is focused on maintaining strong
corporate governance and prioritizing the interests of our
investors. The Board has oversight of our business and affairs,
reviews progress on major strategic initiatives, and receives
progress and status reports on the Firm’s ESG initiatives
throughout the year.
Our Board comprises 16 directors, 10 of whom are independent. Four committees
consisting exclusively of independent directors exercise oversight of our operations
and initiatives. These committees include Audit, Governance and Nominating,
Management Resources and Compensation, and Risk Management. Our Board
conducts annual reviews of our Board charters, which outline the responsibilities of
the Board and committees.
We believe that our business benefits from diversity in backgrounds, experiences
and perspectives. We work to ensure that our Board of Directors includes
individuals with diverse business expertise and international experience, and who
are representative of the communities in which we operate in terms of gender and
ethnic diversity. Our Board Diversity Policy drives progress toward our goals and
underscores our commitment to building a diverse Board. As of 2022, 38% of
Board of Directors positions are held by women, and 60% of our independent
directors are women.
ADDITIONAL INFORMATION
Board of Directors Charter
Board Diversity Policy
Charter of Expectations for Directors
Board Position Descriptions
44
GOVERNANCE
2021 ESG REPORT | Brookfield Asset Management
ESG Governance
Brookfield’s Board of Directors, through its Governance and
Nominating Committee, has ultimate oversight of Brookfield’s ESG
strategy and receives regular updates on the company’s ESG initiatives
throughout the year. Each aspect of ESG is overseen by select senior
executives from Brookfield and each of our business groups, who are
charged with driving ESG initiatives based on our business imperatives,
industry developments and best practices, in each case supported by
asset management professionals from each of these constituencies.
45
GOVERNANCE
2021 ESG REPORT | Brookfield Asset Management
Business Ethics
Strong ethical practices are core to our operating philosophy. Honesty, integrity and respect
are important elements of our Code of Business Conduct and Ethics (Code of Conduct).
We conduct our activities to comply with all
applicable legal and regulatory requirements, and in
accordance with our Code of Conduct. Our Code of
Conduct applies to all Brookfield directors, officers,
employees and temporary workers, including our
wholly owned subsidiaries and any other controlled
affiliates of Brookfield.
Our Code of Conduct principles include:
 Acting responsibly in our dealings with
stakeholders;
 Protecting the Firm’s assets, resources and data;
 Managing conflicts of interest;
 Providing a positive work environment for our
employees;
 Ensuring accuracy of books and records and
public disclosures; and
 Complying with laws, rules, regulations and
internal policies.
The Board annually reviews the Code of Conduct and
considers any necessary changes in the Firm’s
standards and practices. The Risk Management
Committee of the Board monitors compliance with the
Code of Conduct and receives regular reports on any
compliance issues from the Firm’s internal auditors.
Brookfield is committed to an environment where
open and honest communications are the
expectation, not the exception. A significant
component of fostering a positive work environment
is ensuring multiple means by which employees are
able to raise concerns both informally (by fostering a
culture of respect, openness and collaboration), and
formally (through an ethics hotline that permits
anonymous reporting). Our Whistleblowing Program
encourages employees to raise concerns as soon as
possible and to feel safe in doing so.
We have a zero-tolerance approach to bribery,
including facilitation payments. We mandate that all
Brookfield employees complete annual anti-bribery
and corruption (ABC) training and certify their
compliance with our ABC Program. In addition, ABC is
integrated into our investment underwriting, decision
making and execution processes in accordance with
our ABC Policy.
Our ethics hotline, managed by an independent third
party, is available 24 hours a day, seven days a week
to facilitate the anonymous reporting of suspected
unethical, illegal or unsafe behavior.
In addition to Brookfield’s ethics hotline, we require
all portfolio companies in which we have a controlling
interest to adopt our Code of Conduct or ensure that
existing practices are consistent and equivalent in
substance. We also require portfolio companies to
implement an ethics hotline that is accessible to
full-time employees, contractors and temporary
workers, typically within six months of acquisition.
In addition to the ongoing and timely independent
review of employee reports, any significant hotline
reports are brought to the attention of Brookfield’s
senior management and relevant committees of
the Board on a quarterly basis at a minimum.
In 2021, our portfolio companies completed
~90,000 hours of ABC training.
ADDITIONAL INFORMATION
Code of Business Conduct and Ethics
Anti-Bribery and Corruption Program
Anti-Money Laundering and Trade Sanctions Policy
Personal Trading Policy
Business Continuity and Crisis Management Plan
Whistleblowing Policy
Disclosure Policy
Majority Voting Policy
Tax Governance Framework
Clawback Policy
Additional Governance Documents
46
GOVERNANCE
2021 ESG REPORT | Brookfield Asset Management
Human Rights and
Modern Slavery
We are committed to promoting ethical practices and protecting human rights.
HUMAN RIGHTS
Brookfield prioritizes ethical and responsible
business practices including ensuring that our
business operates in a way that respects and
supports the protection of human rights through
working toward:
 Eliminating discrimination in employment;
 Prohibiting modern slavery, including child and
forced labor; and
 Eradicating harassment and physical and mental
abuse in the workplace.
Our core business practices—including contractual
provisions, due diligence processes, training, and
communications—build around our ethics standards.
We require our key suppliers to follow comparable
standards. Our approach and policies continue to
evolve to protect human rights.
MODERN SLAVERY
We have procedures in place designed to prevent
modern slavery, based on the level of risk presented
according to jurisdiction, sector, supplier and other
governance factors.
In 2021, we expanded our U.K. detection and
prevention policies for modern slavery and human
trafficking to cover our entire global footprint. We
have additional policies and procedures in place to
identify and address risks presented by modern
slavery. These policies include our:
 Code of Conduct
 Vendor Management Program, including the
Vendor Code of Conduct
 ESG Due Diligence Guidelines
 ABC Program
 Anti-Money Laundering and Trade Sanctions Policy
 Whistleblowing Program
We also added a separate human rights and modern
slavery risk assessment to our ESG investment due
diligence process, with the objective of mitigating the
risks of modern slavery and human rights violations
for potential investments, including in supply chains.
Where required, we perform deeper due diligence,
working with internal experts and third-party
consultants as needed.
All employees receive modern slavery training during
the onboarding process. Additional training relevant to
applicable regions and roles, particularly in higher-risk
functions such as procurement, is provided.
Portfolio company senior leadership is responsible
for overseeing modern slavery and human rights
governance, and risk mitigation for their individual
businesses. Our Whistleblowing Policy encourages
employees, suppliers and business partners across
the entirety of our operations and global footprint to
report concerns and potential violations.
We produce an annual Modern Slavery and Human
Trafficking Transparency Statement in accordance
with the U.K. Modern Slavery Act 2015 and the
Australian Modern Slavery Act 2018.
We are cognizant of the fact that the risks of modern
slavery and human trafficking are complex and
evolving, and we will continue to work on addressing
these risks in our business.
ADDITIONAL INFORMATION
Whistleblowing Policy
Modern Slavery Statement
47
GOVERNANCE
2021 ESG REPORT | Brookfield Asset Management
BRK Ambiental: Access to Clean Water
and Sanitation Services
BRK Ambiental currently supplies more than 100 municipalities and 16 million
Brazilians with access to clean water and sanitation services, contributing to
the economic and social development of the country, and enhancing the
health of its residents. In 2021, the company underwent construction to
expand water and sewage networks by 831 km and added 82,000 new
connections. The company expects to invest nearly $1.5 billion in the coming
years in its ongoing efforts to expand services, providing thousands of people
with first-time access to clean water and basic sanitation.
The company actively promotes education on the role that clean water and
sanitation play in preventing contamination. Its programs include; (i) a blog
“Saneamento em Pauta” (Sanitation Talk) that provides information in simple
language that is accessible to all readers and has recorded over 3 million visits
since 2019; (ii) the “Busque por Prevenção” (Search for Prevention) campaign
focused on preventing the spread of dengue fever, a mosquito-borne disease
associated with the lack of basic sanitation and; (iii) partnership with “Instituto
Ayrton Senna” which involves training for 45,000 public school teachers.
BROOKFIELD PRIVATE EQUITY
48
GOVERNANCE
2021 ESG REPORT | Brookfield Asset Management
Responsible Contracting
At Brookfield, we believe in providing our employees with the
tools and resources they need to deliver high-quality products.
Our business groups’ responsible contractor policies outline the procedures
and requirements for selecting contractors and subcontractors (collectively,
“contractors”) for required services, including construction, repair and
maintenance projects at our portfolio companies. Individual business groups are
responsible for selecting contractors in accordance with the following guidelines:
 Consideration of cost, competitive risk-adjusted returns and other factors,
including demonstrated skill, experience, dependability and safety record
 Compliance with all applicable local, state and national laws
 Provision of high-quality services, including payment of fair wages and fair
benefits based on local market factors
 Assurance that all workers and contractors retained by such companies
are properly trained and equipped, and perform their work in a safe and
efficient manner
 Avoidance of working with any contractors that are currently debarred
57,000+
unionized full-time operating employees
49
GOVERNANCE
2021 ESG REPORT | Brookfield Asset Management
Audit Oversight
Brookfield’s Audit Committee of the Board monitors the systems and
procedures for financial reporting and associated internal controls, as well as
the independence, experience, qualifications and performance of our
internal and external auditors. The Audit Committee reviews certain public
disclosure documents before their approval by the full Board and release to
the public, such as Brookfield’s quarterly and annual financial statements and
management’s discussion and analysis. The Audit Committee meets regularly
in private session with Brookfield’s internal and external auditors, without
management present, to discuss and review specific issues as appropriate.
ADDITIONAL INFORMATION
Audit Committee Charter
Executive Compensation
Brookfield’s approach to executive compensation is designed to reinforce long-term
stewardship of the business in line with our goal of creating exceptional value for our
shareholders and investors. The majority of our executives’ total compensation is
awarded in the form of long-term compensation, which vests over a five-year period
in arrears. This practice supports a strong alignment of interests between
management and investors. The Board-level Management Resources and
Compensation Committee oversees risks related to Brookfield’s management
resource planning. Since 2012, Brookfield has asked shareholders to cast an advisory
vote on the Firm’s approach to executive compensation on an annual basis (a
“Say-on-Pay” resolution), the results of which the Board and the Management
Resources and Compensation Committee consider when reviewing compensation
policies and procedures, and when making decisions. Our executive compensation
program is designed to reward only consistent performance over the long term.
ADDITIONAL INFORMATION
Statement of Corporate Governance Practices
Say-on-Pay Policy
~85%
of the value of the annual total compensation for our senior
leadership team is received under our long-term plans
~70%
of the value of the annual total compensation for our
managing partners is received under our long-term plans
50
GOVERNANCE
2021 ESG REPORT | Brookfield Asset Management
Data Privacy and Security
We have a responsibility to our stakeholders to protect their personal data.
DATA PRIVACY
Brookfield’s data protocols comply with all local and
national regulatory requirements, including the
European General Data Protection Regulation (GDPR)
and the California Consumer Protection Act (CCPA),
the requirements of which are included in our global
data protection policy.
Our data protection and cybersecurity due diligence
checklist ensures that our management of personal
information complies with legal and regulatory
requirements. The checklist includes due diligence
markers that seek to ensure fair processing,
international transfers, data processors and security
measures to mitigate a possible personal data breach.
EMPLOYEE AWARENESS
Employees are required to attend regular data
protection awareness training, which covers:
 The type of information Brookfield possesses;
 The importance of using—and retaining—this
information only for the business purpose
intended; and
 How to secure this information.
Brookfield employees are required to comply with all
applicable data protection and privacy laws. An
incident of employee non-compliance with our policy
or unauthorized use or disclosure of confidential
information may result in disciplinary action up to,
and including, termination of employment.
CYBERSECURITY
Our data security program, overseen by our Chief
Information Security Officer, ensures the security
of both Brookfield’s data and that of our
shareholders and other stakeholders. Our policies
and procedures cover topics including security
governance, security awareness, employee training,
relevant access and end-point security, vulnerability
management, penetration testing, security
monitoring and incident response.
The Board’s Information Advisory Steering Committee
oversees our cybersecurity functions and ensures that
our program aligns with industry best practices and
meets a high standard across all our businesses. We
use automated technologies to optimize our security
risk detection and response capabilities, in addition to
access controls and anti-malware protections.
Our auditing and cybersecurity practices align with
the National Institute of Standards and Technology
(NIST) Cybersecurity Framework. We review and
update our cybersecurity program annually and
conduct regular external-party assessments of our
program maturity based on the NIST Cybersecurity
Framework. We also regularly engage with third-party
assessors to evaluate the strength of our program
through penetration and/or ethical hacking
exercises. All employees regularly undergo
mandatory continuing cybersecurity training.
Employees in higher-risk functions receive additional
training and cybersecurity awareness education.
Audits, cybersecurity simulations and employee
testing results indicate that our program is effective
in protecting our stakeholders’ information.
In 2021, the results of our NIST Cybersecurity
Maturity Assessment confirmed the rigor and
effectiveness of our program. In 2022, we undertook
initiatives to further enhance our data protection and
threat-intelligence capabilities, and to improve our
processes for third-party risk management. Finally,
in addition to continued mandatory cybersecurity
education for all employees, we enhanced our
phishing simulations to include social engineering.
~98,000
cybersecurity training hours provided across
Brookfield and its portfolio companies
51
OUR PEOPLE
2021 ESG REPORT | Brookfield Asset Management
KPI Appendices
Comprehensive Topic Review and Analysis
Key Performance Metrics
GRI Content Index
SASB Index
52
KPI APPENDICES
2021 ESG REPORT | Brookfield Asset Management
Comprehensive Topic Review and Analysis
The following table sets out the 16 topics that we believe are material to our approach to our ESG principles and policies:
TOPIC DESCRIPTION
OUR INVESTMENT APPROACH
ESG Integration into Our Investment Process Incorporating ESG factors into investment decision-making and valuation, modeling, portfolio construction and engagement with
portfolio companies
Proxy Voting Guidelines and Stewardship and
Engagement Procedures
Managing our investment activities, including proxy voting, in the best interests of our investors, with the goal of protecting and
enhancing the long-term value of our investments
Systemic Risk Management Integrating financial and ESG-related due diligence, and risk and opportunity management into our overall risk management program
Sustainable Finance Offering our investors opportunities to contribute to a sustainable market economy and support adaptation to global climate change
challenges through sustainable investments
BUILDING A BETTER WORLD
Climate Change Strategy GHG Emissions: Working to reach net-zero emissions by 2050 or sooner by reducing our carbon emissions and accelerating the
transition to a net-zero carbon economy through our renewable power business and global transition investment strategy
Clean Energy: Accelerating the transition to a low-carbon economy through Brookfield’s renewable power operations
Green Building: Prioritizing the investment in, and development of, energy-efficient and responsible buildings and businesses
Water and Waste Measuring Brookfield’s water and waste usage, with the objective of conserving water and reducing waste
OUR PEOPLE
Human Capital Development Attracting people aligned with our culture and providing them with opportunities to develop to achieve high engagement and strong
retention, and facilitate smooth succession
Diversity and Inclusion Maintaining a work environment that benefits from different perspectives and reinforces our culture of long-term focus, aligned
interests and collaboration
Occupational Health and Safety Ensuring the health and safety of our workforce via both physical and mental health policies, goals and programs
53
KPI APPENDICES
2021 ESG REPORT | Brookfield Asset Management
TOPIC DESCRIPTION
CORPORATE GOVERNANCE AND ETHICS
Board Effectiveness Monitoring the effectiveness of Brookfield’s Boards through meetings, self-assessments, tracking attendance, training and education,
and other initiatives
Board Skills and Oversight Ensuring the Board possesses the relevant experience, expertise and skills to oversee and support Brookfield in its endeavors
Business Ethics Conducting Brookfield’s business ethically and responsibly within operations as well as business partnerships, including through the
Firm’s values, code of conduct, policies, and processes
Human Rights Ensuring that Brookfield conducts business in a manner that respects and supports the protection of human rights
Responsible Contracting Engaging contractors that conduct their business in a responsible manner as outlined in Brookfield’s responsible contractor policies
Audit Oversight Monitoring the effectiveness and compliance of the Board, management and portfolio companies
Executive Compensation Reinforcing long-term stewardship of the business through our approach to executive compensation
Data Privacy and Security Protecting the Firm, its assets and its data from data privacy threats through strong practices that are in compliance with evolving
regulatory requirements across our asset management operations
54
KPI APPENDICES
2021 ESG REPORT | Brookfield Asset Management
Key Performance Metrics
Diversity Metrics
BROOKFIELD ASSET MANAGEMENT METRICS UNIT TREND 2021 2020 2019
GENDER DIVERSITY
Full-Time Employees FTE up 2,380 1,854 1,615
Female Full-Time Employees % up 46% 45% 45%
Female SVPs and Above % up 27% 22% 16%
Female Board Directors (Full) % up 38% 31% 25%
Female Board Directors (Independent) % up 60% 50% 40%
ETHNIC DIVERSITY GLOBALLY[9]
White % n.a. 52% n.a. n.a.
Asian % n.a. 28% n.a. n.a.
Black % n.a. 4% n.a. n.a.
Hispanic/LatinX % n.a. 3% n.a. n.a.
Two or More Races/Other % n.a. 7% n.a. n.a.
Did Not Respond or Declined to Self-Identify % n.a. 6% n.a. n.a.
ETHNIC DIVERSITY UNDERREPRESENTED GROUPS[10]
Full-Time Employees % n.a. 39% n.a. n.a.
SVPs and Above % n.a. 22% n.a. n.a.
Investment Team % n.a. 29% n.a. n.a.
Operations Team % n.a. 44% n.a. n.a.
9 During 2021 we launched a self-identification process. This is based on our representation as of March 31, 2022.
10 In countries where we have over 100 employees, based on those who self identified (U.S., Canada, Australia, the U.K. and Brazil).
55
KPI APPENDICES
2021 ESG REPORT | Brookfield Asset Management
Environmental Metrics
BROOKFIELD ASSET MANAGEMENT METRICS UNIT TREND 2021 2020 2019
ORGANIZATION
Offices Reported[11] # down 51 54 55
Employee Headcount for In-Scope Offices FTE up 2,555 2,107 1,879
Annual Revenue $M up 75,731 62,752 67,826
Operational Square Footage sq. ft. up 634,824 519,627 452,029
GREENHOUSE GAS EMISSIONS[12]
Scope 1 Direct mtCO2
e up 429 427 416
Scope 2 Indirect (Market-Based) mtCO2
e up 1,925 1,689 1,852
Scope 2 Indirect (Location-Based) mtCO2
e up 2,039 1,728 1,743
Scope 3 Category 6: Business Air Travel mtCO2
e up 2,646 1,165 4,527
ENERGY
Direct Fuel Combustion MWh up 1,999 1,869 1,889
Diesel Fuel % n.a. 1% 1% 1%
Natural Gas % n.a. 99% 99% 99%
Purchased Energy MWh up 7,528 6,113 6,113
Chilled Water % n.a. 12% 12% 9%
Heat/Steam % n.a. 22% 18% 13%
Electricity % n.a. 65% 70% 78%
WATER
Water Consumption m3 up 30,435 26,561 35,781
WASTE
Business Waste metric tons up 151 125 329
Recycled Material % n.a. 50% 50% 46%
Recycled E-Waste[13] % n.a. 100% 100% 100%
11 The scope of offices reported was expanded in 2021 to cover all BAM offices. This ensures alignment with SBTi and GHG Protocol guidance requiring at least 95% of operational emissions to be captured, and the historical years back to 2019 have been
updated to reflect this change. The decrease in office count from 2019 to 2021 is related to office consolidation and/or closure.
12 GHG emissions were measured consistent with the guidelines set out by the GHG Protocol.
13 E-waste volumes vary based on new technology deployment and collection in a given year; however, our target is to recycle 100% of e-waste created.
Actual values were included where available; in some instances, data estimates were calculated based on internal and/or industry-average data, in line with leading industry guidance.
56
KPI APPENDICES
2021 ESG REPORT | Brookfield Asset Management
GRI Content Index
DISCLOSURE NUMBER DISCLOSURE TITLE LOCATION/EXPLANATION
GRI 102: GENERAL DISCLOSURES
Organizational Profile
102-1 Name of the organization Brookfield Asset Management Inc.
102-2 Activities, brands, products, and services Annual Report, p. 25–40
102-3 Location of headquarters Toronto, Canada
102-4 Location of operations Annual Report, p. 176
102-5 Ownership and legal form Annual Report, p. 22
102-6 Markets served Brookfield at a Glance
102-7 Scale of the organization Brookfield at a Glance
102-8 Information on employees and other workers Diversity, Equity and Inclusion
102-9 Supply chain Our supply chain for the Firm is diverse and global, reflecting individual procurement needs for our
various locations.
102-10 Significant changes to the organization and supply chain None
102-11 Precautionary principle or approach Systemic Risk Management
102-12 External initiatives ESG Affiliations and Partnerships
102-13 Memberships of associations ESG Affiliations and Partnerships
Strategy
102-14 Statement from senior decision-maker Letter to Stakeholders
102-15 Key impacts, risks, and opportunities Comprehensive Topic Review and Analysis
Ethics and Integrity
102-16 Values, principles, standards, and norms of behavior Corporate Governance and Ethics
102-17 Mechanisms for advice and concerns about ethics Whistleblowing Policy
57
KPI APPENDICES
2021 ESG REPORT | Brookfield Asset Management
DISCLOSURE NUMBER DISCLOSURE TITLE LOCATION/EXPLANATION
Governance Structure
102-18 Governance structure Management Information Circular, p. 26–44
Stakeholder Engagement
102-40 List of stakeholder groups Stakeholder Engagement
102-41 Collective bargaining agreements We do not have collective bargaining agreements at the Firm level.
102-42 Identifying and selecting stakeholders Stakeholder Engagement
102-43 Approach to stakeholder engagement Stakeholder Engagement
102-44 Key topics and concerns raised Stakeholder Engagement
Reporting Practices
102-45 Entities included in the consolidated financial statements Annual Report, p. 22
102-46 Defining report content and topic Boundaries About This Report
102-47 List of material topics Comprehensive Topic Review and Analysis
102-48 Restatements of information None
102-49 Changes in reporting This is the company’s second report published in accordance with the GRI Standards: Core option.
102-50 Reporting period January 1, 2021 through December 31, 2021
102-51 Date of most recent report June 2021
102-52 Reporting cycle Annual
102-53 Contact point for questions regarding the report esg@brookfield.com
102-54 Claims of reporting in accordance with the GRI Standards This report has been prepared in accordance with the GRI Standards: Core option.
102-55 GRI content index This document represents the company’s content index.
102-56 Policy/practice for external assurance Brookfield is not seeking external assurance for this year’s report.
GRI 200: ECONOMIC DISCLOSURES
GRI 201: Economic Performance
103-1 Explanation of the material topic and its Boundary Annual Report, p.18–20
103-2 The management approach and its components Annual Report, p.18–20
58
KPI APPENDICES
2021 ESG REPORT | Brookfield Asset Management
DISCLOSURE NUMBER DISCLOSURE TITLE LOCATION/EXPLANATION
103-3 Evaluation of the management approach Annual Report, p.18–20
201-1 Direct economic value generated and distributed Annual Report, p. 18–20, 44
201-2 Financial implications and other risks and opportunities
due to climate change
Climate Change Strategy
GRI 205: Anti-Corruption
103-1 Explanation of the material topic and its Boundary Corporate Governance and Ethics
103-2 The management approach and its components Corporate Governance and Ethics
103-3 Evaluation of the management approach Corporate Governance and Ethics
205-2 Communication and training about anti-corruption
policies and procedures
Corporate Governance and Ethics
GRI 300: ENVIRONMENTAL DISCLOSURES
GRI 302: Energy
103-1 Explanation of the material topic and its Boundary Climate Change Strategy
103-2 The management approach and its components Climate Change Strategy
103-3 Evaluation of the management approach Climate Change Strategy
302-1 Energy Consumption within the organization Key Performance Metrics
GRI 303: Water and Effluents
103-1 Explanation of the material topic and its Boundary Water and Waste
103-2 The management approach and its components Water and Waste
103-3 Evaluation of the management approach Water and Waste
303-5 Water consumption Key Performance Metrics
GRI 305: Emissions
103-1 Explanation of the material topic and its Boundary Climate Change Strategy
103-2 The management approach and its components Climate Change Strategy
103-3 Evaluation of the management approach Climate Change Strategy
305-1 Direct (Scope 1) GHG emissions Key Performance Metrics
59
KPI APPENDICES
2021 ESG REPORT | Brookfield Asset Management
DISCLOSURE NUMBER DISCLOSURE TITLE LOCATION/EXPLANATION
305-2 Energy indirect (Scope 2) GHG emissions Key Performance Metrics
305-3 Other indirect (Scope 3) GHG emissions Key Performance Metrics
305-5 Reduction of GHG emissions Climate Change Strategy
GRI 306: Effluents and Waste
103-1 Explanation of the material topic and its Boundary Water and Waste
103-2 The management approach and its components Water and Waste
103-3 Evaluation of the management approach Water and Waste
306-2 Waste by type and disposal method Key Performance Metrics
BAM KPI Waste diverted from landfill Key Performance Metrics
GRI 400: SOCIAL DISCLOSURES
GRI 401: Employment
103-1 Explanation of the material topic and its Boundary Human Capital Development
103-2 The management approach and its components Human Capital Development
103-3 Evaluation of the management approach Human Capital Development
BAM KPI % of global investment team offered transfer stretch
opportunities over 5 years at our current pace
Human Capital Development
GRI 403: Occupational Health and Safety
103-1 Explanation of the material topic and its Boundary Occupational Health and Safety
103-2 The management approach and its components Occupational Health and Safety
103-3 Evaluation of the management approach Occupational Health and Safety
403-1 Occupational health and safety management system Occupational Health and Safety
403-2 Hazard identification, risk assessment, and incident
investigation
Occupational Health and Safety
403-3 Occupational health services Occupational Health and Safety
403-4 Worker participation, consultation, and communication
on occupational health and safety
Occupational Health and Safety
403-5 Worker training on occupational health and safety Occupational Health and Safety
60
KPI APPENDICES
2021 ESG REPORT | Brookfield Asset Management
DISCLOSURE NUMBER DISCLOSURE TITLE LOCATION/EXPLANATION
403-6 Promotion of worker health Occupational Health and Safety
403-7 Prevention and mitigation of occupational health and
safety impacts directly linked by business relationships
Occupational Health and Safety
403-9 Work-related injuries Occupational Health and Safety
403-10 Work-related ill health Occupational Health and Safety
GRI 405: Diversity and Equal Opportunity
103-1 Explanation of the material topic and its Boundary Diversity, Equity and Inclusion
103-2 The management approach and its components Diversity, Equity and Inclusion
103-3 Evaluation of the management approach Diversity, Equity and Inclusion
405-1 Diversity of governance bodies and employees Diversity, Equity and Inclusion
BAM KPI % of investment professionals who are women Diversity, Equity and Inclusion
BAM KPI Ethnic diversity of full-time employees Diversity, Equity and Inclusion
GRI 412: Human Rights Assessment
103-1 Explanation of the material topic and its Boundary Human Rights and Modern Slavery
Responsible Contracting
103-2 The management approach and its components Human Rights and Modern Slavery
Responsible Contracting
103-3 Evaluation of the management approach Human Rights and Modern Slavery
Responsible Contracting
412-2 Employee training on human rights policies or procedures Human Rights and Modern Slavery
GRI 418: Customer Privacy
103-1 Explanation of the material topic and its Boundary Data Privacy and Security
103-2 The management approach and its components Data Privacy and Security
103-3 Evaluation of the management approach Data Privacy and Security
BAM KPI Employee training on Data Privacy and Security Data Privacy and Security
61
KPI APPENDICES
2021 ESG REPORT | Brookfield Asset Management
SASB Index
As part of our ongoing commitment to transparency, we have included the following disclosure under the Sustainability Accounting
Standards Board (SASB) standards for the industries that are relevant to us: Asset Management and Custody Activities.[14]
SASB STANDARD DESCRIPTION CODE RESPONSE
ACCOUNTING METRICS
Transparent Information
& Fair Advice for Customers
(1) Number and (2) percentage of covered employees
with a record of investment-related investigations,
consumer-initiated complaints, private civil litigations, or
other regulatory proceedings
FN-AC-270a.1 Brookfield discloses any material legal
proceedings in its Annual Report
Total amount of monetary losses as a result of
legal proceedings associated with marketing and
communication of financial product-related information
to new and returning customers
FN-AC-270a.2 Brookfield discloses any monetary losses from
material legal proceedings in its Annual Report
Description of approach to informing customers about
products and services
FN-AC-270a.3 Stakeholder Engagement
Please see pages 19–21 of our Annual Report.
Employee Diversity & Inclusion Percentage of gender and racial/ethnic group
representation for (1) executive management,
(2) non-executive management, (3) professionals
and (4) all other employees
FN-AC-330a.1 Diversity, Equity and Inclusion
Incorporation of Environmental,
Social and Governance
Factors in Investment
Management & Advisory
Amount of assets under management, by asset class,
that employ (1) integration of environmental, social
and governance (ESG) issues, (2) sustainability themed
investing and (3) screening
FN-AC-410a.1 (1) $720 billion[15]
(2) $15 billion
(3) $0
Description of approach to incorporation of
environmental, social and governance (ESG) factors
in investment and/or wealth management processes
and strategies
FN-AC-410a.2 ESG Integration Into Our Investment Process
Description of proxy voting and investee engagement
policies and procedures
FN-AC-410a.3 Proxy Voting
14 The SASB Index does not incorporate Oaktree Capital, except for total AUM figure of $688 billion.
15 Total AUM as of Q1 2022
62
KPI APPENDICES
2021 ESG REPORT | Brookfield Asset Management
SASB STANDARD DESCRIPTION CODE RESPONSE
Business Ethics Total amount of monetary losses as a result of legal
proceedings associated with fraud, insider trading, anti-trust,
anti-competitive behavior, market manipulation, malpractice,
or other related financial industry laws or regulations
FN-AC-510a.1 Brookfield discloses any monetary losses from
material legal proceedings in its Annual Report.
Description of whistle-blower policies and procedures FN-AC-510a.2 Whistleblower Policy
Systemic Risk Management Percentage of open-end fund assets under management
by category of liquidity classification
FN-AC-550a.1 (1) 31% highly liquid open-end fund assets
(2) 0% moderately liquid open-end fund assets
(3) 0% less liquid open-end fund assets
(4) 69% illiquid open-end fund assets
Description of approach to incorporation of liquidity
risk management programs into portfolio strategy and
redemption risk management
FN-AC-550a.2 Systemic Risk Management
Total exposure to securities financing transactions FN-AC-550a.3 $0
Net exposure to written credit derivatives FN-AC-550a.4 $0
ACTIVITY METRICS
(1) Total registered and (2) total unregistered assets under management (AUM) FN-AC-000.A (1) $4 billion
(2) $552 billion
Total assets under custody and supervision FN-AC-000.B $688 billion
NOTICE
The information contained herein covers the time period beginning on January 1, 2021, and ending on December 31, 2021, unless otherwise indicated. The information contained herein is intended solely for informational purposes and is not intended to, and
does not constitute, an offer or solicitation to sell or a solicitation of an offer to buy any security, product, or service (nor shall any security, product, or service be offered or sold) in any jurisdiction in which Brookfield is not licensed to conduct business and/ or an
offer, solicitation, purchase, or sale would be unavailable or unlawful. Certain information contained in this publication may constitute “forward-looking statements”.
as defined in applicable securities laws. Forward-looking statements include statements that are predictive in nature, depend upon or refer to future events or conditions, and include statements regarding Brookfield’s operations, business, financial condition,
expected financial results, performance, prospects, opportunities, priorities, targets, goals, ongoing objectives, strategies, and outlook. In some cases, forward-looking statements can be identified by terms such as “expects,” “anticipates,” “plans,” “believes,”
“estimates,” “seeks,” “intends,” “targets,” “projects,” “forecasts” or negative versions thereof, or future or conditional verbs such as “may,” “will,” “should,” “would” and “could.” Although Brookfield believes that the anticipated future results, performance, or
achievements expressed or implied by the forward-looking statements are based upon reasonable assumptions and expectations in light of information available at the time such statement is or was made, reliance should not be placed on forward-looking
statements because they involve known and unknown risks, uncertainties, and other factors, including Brookfield’s ability to identify, measure, monitor and control risks across Brookfield’s entire business operations, including its portfolio companies, which may
cause the actual results, performance, or achievements to differ materially.
Brookfield undertakes no obligation to update or revise statements or information in this publication, whether as a result of new information, future developments, or otherwise. None of Brookfield, its officers, employees, agents, or affiliates makes any express or
implied representation, warranty or undertaking with respect to the accuracy, reasonableness, or completeness of any of the information contained herein, including without limitation, information obtained from portfolio companies or other third parties. Some
of the information contained herein has been prepared and compiled by the applicable portfolio company and has not necessarily been independently verified by Brookfield. Brookfield does not accept any responsibility for the content of such information and
does not guarantee the accuracy, adequacy or completeness of such information. Impacts of initiatives are estimates that have not been verified by a third party and are not based on any established standards or protocols. They may also reflect the influence of
external factors, such as macroeconomic or industry trends, that are unrelated to the initiative presented. The information contained herein is not intended to address the circumstances of any particular individual or entity and is being provided solely for
informational purposes. The information set forth herein does not purport to be complete. Nothing contained herein should be deemed to be a prediction or projection of Brookfield’s future performance. Except where otherwise indicated herein, the
information provided herein is based on matters as they exist as of the date of preparation and not as of any future date and will not be updated or otherwise revised to reflect information that subsequently becomes available or circumstances existing or
changes occurring after the date hereof. All data as of December 31, 2021, unless otherwise noted.
2021 ESG REPORT
| Brookfield Asset Management 63
brookfield.com
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
            per_device_eval_batch_size=64,
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
        # print('=============[get_model_result]==============')
        example_id = example["id"]
        context = example["context"]
        answers = []
        for feature_index in self.example_to_features[example_id]:
            # print((feature_index,tokenizer.decode(eval_set["input_ids"][feature_index])))
            start_logit = self.start_logits[feature_index]
            end_logit = self.end_logits[feature_index]
            offsets = self.eval_set["offset_mapping"][feature_index]

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
                                "logit_score": start_logit[start_index] + end_logit[end_index],
                                "context": tokenizer.decode(self.eval_set["input_ids"][feature_index], skip_special_tokens=False),
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
        self.eval_set = eval_set
        predictions, _, _ = self.trainer.predict(eval_set)
        self.start_logits, self.end_logits = predictions
        self.example_to_features = collections.defaultdict(list)
        for idx, feature in enumerate(eval_set):
            self.example_to_features[feature["example_id"]].append(idx)
        
        firuge_set = small_eval_set.map(self.get_model_result, batched=False)
        return firuge_set
        
        


model_checkpoint = "/home/linzhisheng/esg/QA/bert-finetuned-esgQA-fine-grained-8-2"
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
    print('preprocess_validation_examples is :{}'.format(max_length))
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
    'text': [input,input,input],
    'path': ['ElectricityPurchased',"EnergyUseTotal", "WasteRecycledTotal"],
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

result_set = question(eval_firuge_set,'What is the data {} about?')
result_set.to_csv('data/figure_about.csv',index=False)
# next question What is the unit of?
result_set = question(eval_firuge_set,'What is the unit of {}?')
result_set.to_csv('data/figure_unit.csv',index=False)
# next question What year?
result_set = question(eval_firuge_set,'What year is {} about?')
result_set.to_csv('data/figure_year.csv',index=False)
