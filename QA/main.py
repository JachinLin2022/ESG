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

    # predicted_answers = []
    
    def __init__(self, checkpoint):
        self.checkpoint = checkpoint
        self.model = AutoModelForQuestionAnswering.from_pretrained(checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        args = TrainingArguments(
            "main_test",
            evaluation_strategy="no",
            save_strategy="epoch",
            learning_rate=2e-5,
            num_train_epochs=3,
            weight_decay=0.01,
            per_device_train_batch_size=6,
            per_device_eval_batch_size=128,
            fp16=True,
            push_to_hub=False,
            save_total_limit=1
        )
        self.trainer = Trainer(
            model=self.model,
            args=args,
            # train_dataset=train_dataset,
            # eval_dataset=eval_set,
            tokenizer=self.tokenizer,

        )
        print('[model init success]')

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

                    answers.append(
                        {
                            "text": context[offsets[start_index][0]: offsets[end_index][1]],
                            "logit_score": start_logit[start_index] + end_logit[end_index],
                            # "origin_text": tokenizer.decode(eval_set["input_ids"][feature_index])
                        }
                    )

                    # print((context[offsets[start_index][0] : offsets[end_index][1]], start_logit[start_index] + end_logit[end_index]))

        # best_answer = max(answers, key=lambda x: x["logit_score"])
        top_20_answers = heapq.nlargest(
            20, answers, key=lambda s: s['logit_score'])
        # predicted_answers.append(
        #     {"id": example_id, "prediction_text": best_answer["text"]})
        example['answer_set'] = top_20_answers
        return example
        # print(top_20_answers)
        
    def get_fiture_set(self, small_eval_set, eval_set):
        print('=============[get_fiture_set]==============')
        predictions, _, _ = self.trainer.predict(eval_set)
        self.start_logits, self.end_logits = predictions
        self.example_to_features = collections.defaultdict(list)
        for idx, feature in enumerate(eval_set):
            self.example_to_features[feature["example_id"]].append(idx)
        
        firuge_set = small_eval_set.map(self.get_model_result, batched=False, num_proc=32)
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


def get_figure_from_model(trainer, eval_set):
    predictions, _, _ = trainer.predict(eval_set)
    start_logits, end_logits = predictions
    # with torch.no_grad():
    #     outputs = trained_model(**batch)

    # start_logits = outputs.start_logits.cpu().numpy()
    # end_logits = outputs.end_logits.cpu().numpy()
    import collections
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(eval_set):
        example_to_features[feature["example_id"]].append(idx)

    import numpy as np
    import heapq
    n_best = 20
    max_answer_length = 30
    predicted_answers = []
    print('start to eval')

    

    firuge_set = small_eval_set.map(get_result, batched=False, num_proc=32)
    return firuge_set


input = '''
2019
6169
Environmental, Social and Governance Report

About the Report 2
Company Profile 2
Core Value of the Group 3
Identification and Communication with Stakeholders 3
I. Employment and Labour Practices 5
Employment 5
Health and Safety 8
Development and Training 10
Labour Standards 11
II. Operating Practices 12
Supply Chain Management 12
Service Responsibility 13
Anti-corruption 15
III. Community Investment 16
Public Welfare Volunteer Activities 16
Social Donation 19
Staff Care 20
IV.Environmental 21
Emissions 21
Use of Resources 22
Environment and Natural Resources 24
CONTENTS
2 CHINA YUHUA EDUCATION CORPORATION LIMITED
ABOUT THE REPORT
SUMMARY
This report is the third environmental, social and governance report (the “ESG Report” or the
“Report”) issued by the Group (as defined below). Unless otherwise stated, this Report should be
read in conjunction with the Corporate Governance Report on pages 58 to 71 of the 2019 annual
report of the Company. This Report will be published on the website of The Stock Exchange of Hong
Kong Limited (the “Stock Exchange”) and on the Group’s website. The ESG Report will be published
annually.
BASIS OF PREPARATION
This ESG Report has been prepared in accordance with the “Environmental, Social and Governance
Reporting Guide” (the “ESG Guide”) as set out in Appendix 27 to the Rules Governing the Listing of
Securities on The Stock Exchange of Hong Kong Limited (the “Listing Rules”)
SCOPE AND EXTENT OF THE REPORT
The data and information referred to in this Report are derived from various files, questionnaires,
records, statistics and research of the Group. This Report covers the period from 1 September 2018
to 31 August 2019 (the “Reporting Period”), which corresponds to the financial year covered in the
2019 annual report.
The policy document, declaration and data set out in this Report cover the Company and its
subsidiaries and consolidated affiliated entities (collectively, the “Group”).
CONTACT INFORMATION
If you have any queries or feedback about this Report and its contents, please contact us at:
• Address: 4/F, Yuhua Education Business Building, Jinhui West Street, Zhengdong New District,
Zhengzhou City, Henan Province, China
• Tel.: +86 371-60673938
• Fax: +86 371-6595070
• E-mail: contact@yuhuachina.com
• Official website: http://www.yuhuachina.com
COMPANY PROFILE
The Group is one of the largest private education groups in China. The Group currently operates 29
schools covering pre-school education, primary education and higher education in central China and
Thailand. In providing K-12 and university education services, the Group not only focuses on helping
students achieve excellent academic performance, but also emphasises the overall coordinated
development of “body quotient, moral intelligence quotient, emotional quotient and intelligence
quotient” of students, dutifully shouldering the responsibility of nurturing social pillars.
ENVIRONMENTAL, SOCIAL AND GOVERNANCE REPORT 2019 3
ABOUT THE REPORT (CONTINUED)
CORE VALUE OF THE GROUP
Adhering to the core value that “the essence of education is love, the essence of love is giving, and
giving is getting”, the Group adheres to the educational concept aiming at “fostering modern talent
with leadership and lifelong learning capabilities and nuturing great minds to contribute to the future
development of the Chinese nation” and provides students with education services that are in line with
the values of the Group. At the same time, the Group’s teachers abide by the principle of “working
hard, educating people with love, cultivating love with love, seeking truth with truth” to cultivate
talented youth with well-rounded and coordinated development.
IDENTIFICATION AND COMMUNICATION WITH STAKEHOLDERS
While managing its schools and advancing its business affairs, the Group also pays attention to the
major issues of interest to shareholders, investors, staff, students, parents, governments, regulatory
authorities, and communities (“Stakeholders”). It opens up multiple channels of communication
and, through the continuous communication with the Stakeholders, is able to develop thorough
understanding of the needs of different Stakeholders and provide appropriate solutions. At the same
time, the Group believes that listening to the opinions of Stakeholders will help the Group to improve
its environmental, social and governance performance comprehensively and objectively so as to better
address the needs of different Stakeholders.
Key Concerns of Stakeholders and the Corresponding Actions
Stakeholder Main focus Communication channels Corporate/Group actions
Shareholders/
Investors
Operating strategy;
Sustainable and stable return
on investment;
Timely information disclosure;
Excellent enterprise image;
and
Operation of enterprise in
compliance with relevant
laws and regulations.
General meeting of
shareholders;
Information disclosure of the
listed company;
Roadshows/conference
calls/meetings;
Media communication
mechanism;
Enquiries via telephone/email;
Investors’ on-site visit; and
Website information
disclosure.
Issue of notice of annual
general meeting and
the resolutions as required
by the Listing Rules;
Timely disclosure of
information about
the Group;
Issue of announcements and
regular reports as required
by the Listing Rules; and
Provision of smooth
communication channels.
Staff Training and career
development space;
Salary and welfare;
Working environment; and
Health and safety protection.
Direct communication;
Physical examination;
Staff activities;
Opinions from staff; and
Staff training.
Providing healthy and safe
working environment;
Setting up a fair promotion
system;
Providing staff with interactive
platform; and
Organising staff activities.
4 CHINA YUHUA EDUCATION CORPORATION LIMITED
ABOUT THE REPORT (CONTINUED)
Stakeholder Main focus Communication channels Corporate/Group actions
Students and
Parents
Educational service quality;
Student information
protection;
Student life care;
Health and safety protection;
Teaching quality; and
Student performance.
Collection of complaints and
feedback;
Maintaining good
communication with
students;
Caring for student life;
Helping families suffering from
difficulties; and
Parents meetings.
Establishing a parent
committee;
Conducting student surveys;
Organising student activities;
Regular physical examination;
Regular parents meetings;
and
Maintaining good
communications.
Government
and
Regulatory
authorities
Operational compliance;
Tax compliance;
Transparent governance; and
Information disclosure and
reporting materials.
Compliance with laws and
regulations;
Routine work report; and
Information disclosure.
Strict compliance with laws
and regulations;
Accurate disclosure of
information;
Tax payment by law; and
Accepting government
supervision.
Community Employment opportunities;
Ecological environment;
Community development; and
Social commonwealth.
Community engagement. Priority hire
of local staff;
Preserving the environment;
and
Organising community
activities.
Media Open information; and
Good media relations.
Information disclosure. Maintaining good
communication; and
Timely disclosure
of information.
ENVIRONMENTAL, SOCIAL AND GOVERNANCE REPORT 2019 5
I. EMPLOYMENT AND LABOUR PRACTICES
Adhering to the teaching principle of “working hard, educating people with love, cultivating love with
love, seeking truth with truth”, the Group earnestly takes the responsibility of educating people and
cultivating pillars for national and social development. The Group strictly obeys relevant laws and
regulations to recruit and hire teachers and staff, and has established a scientific, fair and mature
talent employment mechanism and human resource management system. The Group has attached
great importance to the health and safety issues of staff, teachers and students, and comprehensively
guarantees a safe, stable and comfortable working and learning environment. The Group has also
provided a fair and scientific career development platform for teachers and employees to enhance
their professionalism and teaching ability.
EMPLOYMENT
During the recruiting and hiring process, the Group strictly obeys the Labour Law, the Labour
Contract Law, the Employment Promotion Law, the Education Law, the Teachers Law, the Labour
Dispute Mediation and the Arbitration Law of the People’s Republic of China (“China” or the “PRC”)
and the Labour Protection Act, the Labour Relations Act, the Social Security Act, the Workmen’s
Compensation Act of Thailand, as well as the local labour laws and regulations in the provinces
of Henan, Hunan and Shandong of China. During the Reporting Period, the Group has complied
with all applicable laws and regulations, and has not been subject to warnings, fines, and penalties
for violations of laws or regulations. The Group has not violated laws and regulations related to
recruitment and promotion, compensation and dismissal, working hours, holidays, equal opportunities,
diversity, anti-discrimination and other benefits, as well as receiving any warnings, fines, penalties and
other punitive events.
During the Reporting Period, there were no major updates on the human resource policies and related
hiring procedure. The Group employs staff and carries out recruitment strictly based on the Personnel
Business Process in the Staff Handbook of the Group. The Group treats every candidate fairly
regardless of gender, nationality and age, and provides teachers with fair employment opportunities
and a harmonious working environment. The Group starts initiating the recruitment of graduates every
November by cooperating with “58.com”, “Zhaopin.com” and other well-known recruitment agencies.
In addition, the termination of labour contracts can be divided into three categories: resignation,
dismissal, and natural termination of labour contract. The Group has also provided detailed guidance
on the termination process in the Personnel Business Process. The Group has the right to dismiss
employees in certain circumstances. For instance, the Group may issue the Notice of Termination of
Labour Contract to employees who seriously violated the relevant provisions in the Staff Handbook.
Formal employees can submit a Resignation Report to the head of the department for further approval
30 days in advance.
In terms of working hours, the Group has made detailed guidelines for working hours and attendance
regulations in strict accordance with the Time Management System in the Staff Handbook. The
Group keeps optimising the Group’s working processes and improving staff working efficiency. In the
meantime, the attendance record is used as one of the important standards for staff assessment,
promotion and transfers.
6 CHINA YUHUA EDUCATION CORPORATION LIMITED
I. EMPLOYMENT AND LABOUR PRACTICES (CONTINUED)
In terms of compensation and welfare benefits, the Group guarantees the legal benefits of staff in
accordance with Tentative Provisions on Payment of Wages, Regulations on the Administration of
Housing Fund, Regulation on the Annual Leave with Pay and other laws and regulations. The Group
also aims to meet the overall local requirements of social security policies, including endowment
insurance, medical insurance, maternity insurance, unemployment insurance, critical illness insurance
and other social insurance. In compliance with the Labour Law, the Group ensures that staff can enjoy
all kinds of holidays including public holidays, paid annual leave, sick leave, marriage leave, maternity
leave, etc. The Group also provides relevant benefits to staff during major festivals in China. The
Group also provides free accommodation for teachers and staff of all units in the Group, and regularly
conducts activities to enrich daily life of employees.
Upholding the promotion assessment mechanism of “valuing abilities regardless of educational
background, valuing attitudes regardless of qualification and valuing performance regardless of
certificates”, the Group has enacted a mature, fair and scientific promotion assessment mechanism.
During the Reporting Period, the average age of middle management personnel of the Group was 38
years old. For a long time, the Group has been providing a fair career development platform for all staff
and aims to ensure fair treatment in terms of employment, assessment, promotion, training, etc. Any
discrimination related to religion, gender, age, and ethnicity is strictly forbidden in the Group. During
the Reporting Period, the Group did not have any discrimination incidents.
By the end of the Reporting Period, the aggregate number of staff members in the Group was 8,094,
with a proportion between men and women, which accounted for 36.4% and 63.6%, respectively,
and the staff turnover rate was 9.4%. Due to the characteristics of the education industry, the Group
has a higher proportion of female staff. In compliance with the law, the Group provides statutory
benefits, including maternity leave, marriage leave and breast-feeding leave for female staff, to ensure
that they are not discriminated against or otherwise disadvantaged.
ENVIRONMENTAL, SOCIAL AND GOVERNANCE REPORT 2019 7
I. EMPLOYMENT AND LABOUR PRACTICES (CONTINUED)
EMPLOYMENT INDICATORS
Senior Management
Personnel
Middle Management
Personnel
Ordinary Staff
over 60
30 or below
31−40
41−50
51−60
Male Female
Departed Staff In-service Staff
Number of Staff
Number of Staff
Staff by Gender in 2019
Staff by Employment Type in 2019
Staff by Departure Rate in 2019
0 2,000 4,000 6,000 8,000
2,948 5,146
0 2,000 4,000 6,000 8,000 10,000
761 8,094
Staff by Age in 2019
0 500 1,000 1,500 2,000 2,500 3,000
2,645
2,814
1,417
877
341
0 1,000 2,000 3,000 4,000 5,000 6,000 7,000 8,000
7,565
517
12
8 CHINA YUHUA EDUCATION CORPORATION LIMITED
I. EMPLOYMENT AND LABOUR PRACTICES (CONTINUED)
Henan Province Shandong Province Hunan Province Thailand
Staff by Geographical Region in 2019
4,301
1,812
1,615
366
HEALTH AND SAFETY
The Group strictly obeys the Food Safety Law, the Management Regulation on Student Canteen and
Student Group Meal Hygiene, the Regulations on the Administration of Sanitation in Public Places, the
Law on Prevention and Treatment of Infectious Diseases, the Law on Fire Control of the PRC and the
Food Act and the Public Health Act of Thailand, as well as other relevant laws and regulations in the
PRC and Thailand. The Group has not been punished by warnings, fines, and penalties for violations
of laws or regulations during the Reporting Period. The Group has paid great attention to the health
and safety issues of staff and students. Following the basic principle of “paying attention to prevention,
self-rescue and mutual aid, ensuring safety and reducing losses”, the Group has formulated the Staff
Health and Safety Management System of Yuhua Education Group to ensure the health and safety of
staff and students of the Group. The Group has set up strict safety management rules and guidelines
in terms of fire safety, health management, facilities and equipment management, anti-smoking and
other aspects to practically provide a healthy and safe working and learning environment for teachers
and students.
Fire Safety: The Group has incorporated fire safety into daily management, and has formulated a
fire safety system in accordance with the requirements of the Law on Fire Control. The Group has
set up small fire stations in each of the campuses which are specifically responsible for fire safety
matters on the campuses. In addition, the Group holds fire drills and emergency escape drills every
semester, which helps teachers and students to cope with sudden fire incidents while promoting
fire safety awareness. The Group’s schools organise regular fire drills every year and invite local firefighters to go to schools to educate teachers, staff and students on fire safety knowledge and to
provide guidance during fire drills. During the Reporting Period, a total of 50 fire drills were conducted
at schools of the Group with the participation of a total of 89,926 people.
ENVIRONMENTAL, SOCIAL AND GOVERNANCE REPORT 2019 9
I. EMPLOYMENT AND LABOUR PRACTICES (CONTINUED)
Fire Drills Photos
Health Management in School Areas: In order to improve students’ health standards, the
Group has established and improved health management related policies and systems by clarifying
responsible persons and establishing a regular working procedure. The Group’s subordinate schools
have set March and November as the months of the education and publicity of health, and have
continuously improved the disease prevention and control system, infectious disease isolation system,
physical examination system, and health file management system. In order to provide a healthy and
safe campus environment, the Group has improved the health management level in school areas in
all aspects by standardising the supervision and management of teaching hygiene, environmental
sanitation management system, sanitary inspection system, and canteen sanitary supervision and
management system.
In addition, the Group has also established a complete management system for the management of
facilities and equipment such as air conditioners. Through a scientific and systematic management
system, the Group effectively monitors and manages the operation of facilities and equipment, as well
as maintaining and checking for potential safety hazards. At the same time, the Group has established
a strict anti-smoking management system to ensure a safe and civilised office environment for
employees, which is expected to protect the health of employees, and to maintain a good working,
studying and living environment on the campus.
10 CHINA YUHUA EDUCATION CORPORATION LIMITED
I. EMPLOYMENT AND LABOUR PRACTICES (CONTINUED)
In addition, the safety and health inspection projects which have been set up by the Group include:
safety and health publicity and education; investigation and rectification of hidden safety hazards;
management of dangerous chemicals; canteen food and boiler safety management; police and
security work; medical health management; dormitory safety management; school bus safety
management; and rectification of the campus and surrounding environment, among others. In
response to various health and safety work arrangements, the General Affairs Department of the
Group requires all units to keep relevant records and conduct regular inspections to ensure that staff
and students can work in a safe environment.
During the Reporting Period, there were no work-related injuries or deaths in the Group.
DEVELOPMENT AND TRAINING
The teaching and management abilities of teachers and management staff are directly related to the
teaching quality, management level and brand image of the Group’s schools. Therefore, the Group
has formulated detailed training programs to enhance the knowledge and professional competence of
teachers and management personnel. With the aim of building and passing down experience, training
activities mainly use a case analysis training model and can be divided into three categories: internal
training; external training; and self-training by staff. During the Reporting Period, 100% of the Group’s
staff received such training. Senior and mid-level management personnel completed an average of 48
training hours while other staff members completed an average of 128 training hours.
During the Reporting Period, the major training activities organised by the Group included:
 From 20 March to 20 April 2019, a one-month graduate training event held at the Group’s
headquarters. Approximately 500 people were trained and had completed an average of 160
learning hours.
 On 10 April 2019, the teaching department of the Group organised the head teacher training and
all middle management personnel and head teachers participated in this training. Approximately
600 people were trained and had completed an average of 28 learning hours.
 From 15 August to 20 August 2019, all subordinate schools of the Group organised the summer
teacher training for all teachers. Approximately 2,954 people were trained and completed an
average of 28 learning hours.
 During the Reporting Period, the Group had organised six training activities in association with
personnel management system, education, moral education and other business activities.
Approximately 160 middle management personnel were trained and completed an average of
128 learning hours.
 The “Blue Project” implemented by the Group carries out one-hour training activities for all
young teachers every week, so that senior teachers can teach, help and guide young teachers.
Approximately 800 young teachers were trained and had completed an average of 128 learning
hours.
ENVIRONMENTAL, SOCIAL AND GOVERNANCE REPORT 2019 11
I. EMPLOYMENT AND LABOUR PRACTICES (CONTINUED)
Photos of Staff Training Activities
LABOUR STANDARDS
The Group strictly obeys the Labour Law, the Protection of Minors Law, the Provisions on the
Prohibition of Using Child Labour, the Teachers Law, the Code of Ethics of Teachers in Primary and
Secondary Schools of the PRC and the Labour Protection Act, the Act on Establishment of Labour
Courts and Labour Courts Procedures of Thailand, as well as other relevant laws and regulations
in the PRC and Thailand to recruit and hire staffs that protect the legitimate rights and interests of
teachers and students. The Group prohibits any employment which would constitute child labour
and forced labour, including compulsory labour and improper punitive measures. The Group clearly
stipulates in the recruitment policy and processes that employment of child labour and forced labour
are forbidden. The Group strictly implements the recruitment and hiring procedures in the Staff
Handbook, and carefully checks the identity information of employees before hiring to ensure the
truth and validity of personal information. During the Reporting Period, the Group did not have any
form of compulsory labour or child labour incidents and related complaints. If any violations were
to be detected, the Group would immediately cease any labour activities. Any false documents
would be considered fraudulent and the Group would have the right to terminate the labour contract
immediately.
12 CHINA YUHUA EDUCATION CORPORATION LIMITED
II. OPERATING PRACTICES
SUPPLY CHAIN MANAGEMENT
During the Reporting Period, the Group’s supply chain management system functioned smoothly.
With a comprehensive management system in place, it ensured the procurement needs of subordinate
schools were fulfilled and fully considered the environmental and social risks of suppliers. Major
materials that the Group purchases are office supplies, wooden furniture, iron furniture, electronic
equipment, teaching and tutoring materials, school uniforms and other goods. During the Reporting
Period, the Group had 84 suppliers in total, with whom the Group has maintained multiple years of
cooperation relationships. Out of the 84 suppliers, 4 were from Beijing, 2 from Shandong province, 1
from Liaoning province and the remaining 77 from Henan province.
The Group orders, purchases and distributes necessary materials for daily operations of schools
according to the Supplier Management Operation Manual. In order to standardise material supply
procedures, improve work efficiency, efficiently complete the supply of high quality materials, and
strengthen monitoring and management of suppliers, the Group has formulated the Measures
on Management of Customers of Yuhua Education Group. This is used for conducting scientific
management of the Group’s suppliers, including classification and screening of suppliers,
management of information databases, assessment of suppliers and other aspects. The Group
conducts assessment and rating of suppliers during the annual summer and winter vacations. If
suppliers are found unsuitable, cooperation is terminated in a timely manner.
As an education service provider, the Group is dedicated to creating a safe, hygienic, comfortable and
stable campus environment for teachers and students. In addition to considering the quality, brand
names and qualifications of suppliers and their products, the Group also fully considers environmental
and social risk factors of suppliers when screening suppliers. For example, we require suppliers to
provide environmental impact assessment and quality inspection reports from Henan province when
purchasing uniforms and other materials. We check the qualification certificate on raw material for
products provided by suppliers when purchasing furniture, electrical appliances, teaching equipment,
etc. In terms of supply chain management, the Group also takes measures favourable to creating
environmental and social benefits. Firstly, the Group includes the purchase of materials and approval
process in the enterprise resource planning (ERP) system. The Group also advocates for a paperless
office. In addition, the Group adopts the semi-electronic operation in the process of bidding, and all
kinds of documents are presented in electronic version to reduce the use of paper.
In compliance with the requirements of the Group’s supplier management system, the procurement
department can select suppliers based on historical procurement experience and local market
conditions. Currently, there are 16 suppliers hired in accordance with the above practice, who
mainly supply daily materials procurement. Since such kind of suppliers can meet the procurement
needs of the Group’s schools in the local area and emergent procurement demands, they are good
supplements and good partners in the Group’s supplier system. The Group can also hire suppliers
based on the principle of “Priority on Efficiency and Quality” to satisfy the procurement needs with
small contract value after sufficiently considering the price fluctuation, geographical location, personnel
arrangement, car arrangement, delivery time and other factors. However, the procurement need with
large contract value should be carried out in accordance with the Supplier Management Operation
Manual.
ENVIRONMENTAL, SOCIAL AND GOVERNANCE REPORT 2019 13
II. OPERATING PRACTICES (CONTINUED)
The Group has established a comprehensive monitoring and supervising system for the hiring of
suppliers and materials procurement. The asset management department, general affairs department
and the departments who use materials can provide supervision opinions on the procurement price
and the quality for the procurement department. The asset management department can check
the price and quality of materials through the ERP system. Once detecting any unusual case, the
procurement department should take action immediately and re-evaluate the suppliers promptly
in accordance with the Measures on Management of Customers of Yuhua Education Group. If the
supplier is responsible for the problem, the Group will never cooperate with it any more.
SERVICE RESPONSIBILITY
The Group and its subordinate schools carry out education work in strict accordance with the
Education Law, the Compulsory Education Law, the Higher Education Law, the Non-state Education
Promotion Law, Several Provisions on the Administration of Non-state-operated Colleges and
Universities, the Provisions on the Administration of Students in Regular Institutions of Higher
Education of the PRC and the National Education Act of Thailand, as well as applicable laws suitable
for different school levels, and regulations of other relevant national laws of the PRC and Thailand.
Each campus of the Group has introduced a series of policies and activities to ensure teaching quality:
 University education: The Group’s universities, Zhengzhou Technology and Business University
and Hunan International Economics University, have formulated scientific, systematic and wellestablished teaching management systems and related teaching quality supervision systems
including class observation system, teaching supervision, teaching quality monitoring, course
evaluation and information feedback to ensure the service quality of university education.
Relevant policies and systems enacted by Zhengzhou Technology and Business University
include the Teaching Quality and Monitoring Bulletin, the Teaching Inspection System, the
Class Observation by Administrative Personnel System, the Two-level Supervision Work Plan,
the Identifying Methods for Teaching Quality Evaluation Level, etc. Relevant policies and
systems enacted by Hunan International Economics University include the Daily Teaching
Inspection System, the Class Observation System, the Regulations on Teaching Supervision, the
Implementation Measures for Quality Control of Practical Teaching, etc.
 Primary and secondary school education: Primary and secondary schools mainly adopt the
collective lesson preparation model to ensure the quality of teaching. Primary and secondary
schools organise teaching and research activities twice a week. Classrooms are not locked when
the teacher gives lessons in order to facilitate the supervision of teaching by the academic affairs
office, supervision office and other teachers as well as the observation and learning of other
teachers at the same time. Relevant policies and regulations enacted by the Group’s primary
and secondary schools include the Regulations on the Management of Teaching Practices, the
Class Observation System, the System of Teaching and Research Activities, the Provisions for
Teaching Assessment, etc.
 Kindergarten education: The Group regularly conducts teaching and research activities
including class appraisal, the teaching assistant’s class evaluation, observation classes and other
activities in order to enrich the teaching quality. In addition, the content learned by young children
is assessed and evaluated every month and the results are included in the performance appraisal
standards of teachers.
14 CHINA YUHUA EDUCATION CORPORATION LIMITED
II. OPERATING PRACTICES (CONTINUED)
All subordinate schools of the Group have enacted the policy titled “Identification and Treatment of
Teaching Accident”. For any teacher with deficiencies in teaching quality, schools will talk to and guide
him/her to improve, while including in the relevant performance appraisal to avoid the recurrence of
teaching accidents. To deal with complaints about education services, the Group has set up a special
investigation team and made arrangements for the school leader to communicate with students and
parents and listen carefully to the opinions of parents in order to find the shortcomings and improve
supervision and inspection efforts. During the Reporting Period, the subordinate schools of the Group
did not receive any complaints.
The Group has adopted reasonably effective marketing strategies to attract students and parents.
Major marketing channels include Weibo, WeChat and other social media channels. During the
Reporting Period, the marketing and promotion activities all abided by the Advertising Law and other
laws and regulations.
The Group has introduced a series of policies to ensure the safe, stable and healthy development of
students at campus. For instance, Zhengzhou Technology and Business University has formulated the
Regulations on the Management of Students Safety, the Regulations on the Management of Students
Dormitories, and the Emergency Plans for Fire Safety and Management of Student Apartments,
etc. Hunan International Economics University has formulated the Laboratory Safety Management
Measures and organises security checks regularly. In addition, the Group mainly adopts supervision
and monitoring, returning visits to parents, students’ evaluation and safety education to supervise
and management regular education services in primary and secondary schools. In the meantime, the
Group has obtained timely feedback information and continuously improves the quality of teaching
services. Regarding kindergarten education, the Group mainly adopts supervision and monitoring,
returning visits to parents, safety education and sanitation safety to guarantee the health and safety for
kids during the teaching process.
In terms of knowledge copyright protection, the teaching materials used by the subordinate schools
of the Group are all ordered from authorised publishers, and the Group purchases the teaching
resources website accounts for teachers to ensure that schools at all levels use the educational
resources with copyright. The Group has also formulated the Measures for Morality and Talents
Enhancement Teaching Material Management, the Measures for Intellectual Property Management of
Zhengzhou Technology and Business University and Measures for Patent Management of Zhengzhou
Technology and Business University to ensure that the relevant intellectual property rights are
protected properly.
The Group has formulated the Student File Management Work, the Measures for Archive
Management, the Measures for Student File Management of Zhengzhou Technology and Business
University and other policies to protect the security of personal information. The Group has also
signed non-disclosure agreements with staff that may be involved in the student information safety
and private information generally. The Group also carries out relevant training to instruct teachers and
staff to strictly abide by the obligation to maintain confidentiality and respect the privacy of students.
ENVIRONMENTAL, SOCIAL AND GOVERNANCE REPORT 2019 15
II. OPERATING PRACTICES (CONTINUED)
ANTI-CORRUPTION
The Group strictly obeys the Criminal Law, the Company Law, the Interim Provisions on the Prohibition
of Commercial Bribery, the Anti-Money Laundering Law, the General Principles of Civil Law, the AntiUnfair Competition Law, the Contract Law of the PRC and the New Anti-Corruption Law and the
Criminal Code of Thailand as well as other laws and regulations of the PRC and Thailand to prevent
bribery, extortion, fraud and money laundering and other corrupted incidents.
In order to regulate the professional behaviour of staff, the Group strictly obeys the relevant laws,
industry norms and standards of professional ethics, and rules and regulations of the Group. The
Group has also formulated the Measures for Anti-embezzlement and Reporting Management
Mechanism to prevent bribery, extortion, fraud, money laundering and other types of embezzlement.
In addition, the Group requires any staff involved in economic activities to sign and abide by the
Letter of Commitment of Honesty and Self-discipline. Staff members are held accountable if any
violations are detected. Further, all suppliers, service providers and contractors which have business
relations with the Group must also sign the Anti-Commercial Bribery Agreement before establishing
the cooperative relations. The human resources department, legal department and internal control
department of the Group also conduct training to strengthen the knowledge of staff members in
relation to bribery, extortion, fraud, money laundering and other illegal activities in order to establish
the correct values and strengthen the ability of staff to identify and distinguish legal and illegal, honest
and dishonest, and moral act and immoral acts.
During the Reporting Period, there were no significant changes to the illegal acts and related
enforcement and monitoring measures formulated by the Group. Major measures are as follows:
 setting up the reporting telephone and mailbox as the channel to report actual or suspected
embezzlement cases for which the internal control department is responsible for accepting,
retaining and handling reports;
 the internal control department may also carry out random checks on work procedures and
results of departments engaged in economic activities;
 the finance department regularly examines economic activities and delivers suspected cases of
embezzlement to the internal control department for investigation;
 the asset department checks the work of departments with the ability to purchase through
market research and delivers suspected cases of embezzlement to the internal control
department for investigation; and
 for any staff who engages in embezzlement whether or not amounting to a criminal offence, the
internal control department will recommend company management to impose corresponding
internal economic and administrative disciplinary punishments according to the regulations, and,
should the staff member possibly be in violation of the law, the internal control department will
transfer the case to the relevant authorities.
During the Reporting Period, the Group did not have any bribery, extortion, fraud, money laundering or
other embezzlement cases.
16 CHINA YUHUA EDUCATION CORPORATION LIMITED
III. COMMUNITY INVESTMENT
The Group actively fulfils its corporate social responsibility by participating in the cause of public
welfare and community development. The Group gives full play to its own strengths including by
actively participating in all kinds of community activities and organising teachers and students to
learn through community education, humanistic care, culture and art, urban construction and other
activities. Further, the Group fully reflects staff care in providing jobs and a good working environment,
including providing diversified training and promotion opportunities. In addition, the Group’s schools
attach great importance to the ideological and moral education of students and strive to cultivate
good moral character and a strong sense of social responsibility. Schools regularly communicate with
parents on education methods to create a harmonious family environment.
During the Reporting Period, the Group’s subordinate schools actively took social responsibility
through various channels and means. The Group has carried out poverty alleviation activities in poor
rural areas, and participated in social welfare activities by providing funds, materials and manpower.
Student from subordinate universities actively participated in aid education and continuously raised
their awareness of social responsibility. Subordinate primary and secondary schools not only actively
participated in the construction of civilised cities and volunteering activities, but also organised
lectures and practical activities related to social responsibility to promote social responsibility
awareness. Also, a variety of community practical activities have been carried out at subordinate
kindergartens to cultivate and develop the sense of social responsibility for young children through
personal experience.
PUBLIC WELFARE VOLUNTEER ACTIVITIES
In the Reporting Period, the Group’s schools have made full use of their advantages and have
carried out many public welfare activities. During the Reporting Period, the major social public welfare
activities held by each school of the Group are:
Schools Major social public welfare activities
The Group  Participated in the poverty alleviation activities in Ansheng Village of
Anyang, Fangwa Village of Xinyang.
 Participated in the poverty alleviation activities through precise
education in the Primary School of Nanguanzhuang of Wuzhi County,
Jiaozuo City.
Zhengzhou Technology and
Business University
 Organised and participated in the voluntary aid education in the
Hegang Primary School, Hegang Village, Zhangqiao Town, Fuling
County.
Hunan International
Economics University
 In the financial year of 2019, the school launched a number of public
welfare activities such as visiting the nursing home, helping the
disabled through the Sunshine Campus, voluntary blood donation,
aid education in summer break, etc. Nearly 5,000 students have
participated in the above activities.
ENVIRONMENTAL, SOCIAL AND GOVERNANCE REPORT 2019 17
III. COMMUNITY INVESTMENT (CONTINUED)
Schools Major social public welfare activities
Kaifeng Yuhua
Elite School
 In September 2018, the school participated in the volunteer activities
of reading meetings of Liuxiaolingtong.
 In November 2018, the school launched an environment protection
activity. Volunteers picked up the trash on campus and surrounding
areas.
 In December 2018, the school participated in the volunteer activities
in Kaifeng Library. During the same month, the primary school sector
launched the volunteer activity of learning Leifeng.
 In February 2019, the school participated in the public welfare activity
of “New Green Fashion, Yuhua Charity Walk”.
 In March 2019, the school participated in the volunteer activity at the
Day of Leifeng.
 In April 2019, the school participated in the environment protection
action in Bianxi Lake.
 In May 2019, the school participated in the volunteer activity at the
Youth Day.
 In June 2019, the school participated in the volunteer activity
organised by School Youth League Committee.
 In August 2019, the primary school sector organised the activity of
clean hometown and beautify the environment. Volunteers helped the
sanitation workers at the fifth street and cleaned road railing.
Jiaozuo Yuhua
Elite School
 In October 2018, the school organised students to conduct
compulsory aid education in child welfare centre.
 In November 2018, the school organised condolences activity in the
Second Charity Hospital of Jiaozuo. During the same month, the
school organised the donation activity of “Thanksgiving and Walk with
Love”.
 From January 2019 to August 2019, the school organised “Zhi Zhi
Shuang Fu” activity, providing aid education service to areas with
scarce education resources.
 In March 2019, Leifeng Association of Jiaozuo City held a large-scale
public welfare activity of leaning Leifeng.
 In April 2019, volunteers of young teachers conducted condolence
activity to Jiaozuo Army Division and delivered the common sense of
sports emergency.
18 CHINA YUHUA EDUCATION CORPORATION LIMITED
III. COMMUNITY INVESTMENT (CONTINUED)
Schools Major social public welfare activities
Jiyuan Yuhua
Elite School
 In the financial year of 2019, the school organised the activity
of Condolences to the Cutest Person Around. Volunteers took
condolences for sanitation workers and traffic police. The school
also participated in the Meteorological Festival of Jiyuan, Opening
Ceremony of World Environment Day, Chinese New Year Party for
Children, etc.
Luohe Yuhua
Elite School
 In October 2018, the school organised students to donate books to
children from poor areas.
 On Tomb-Sweeping Day in 2019, the school sent student
representatives to Luohe Revolutionary Martyrs Cemetery to carry out
the education activity of “Worshiping the Heroes in Qingming, Casting
Chinese Souls Together”, cherishing the heroes and inherited red
spirit.
 On Labour Day in 2019, the school organised students to walk on
streets to take condolences to hard-working people.
Kindergarten Headquarter  In April 2019, the school conducted the donation activity of “Hand by
Hand” with Zhaoji Primary School, Xitao Village, Zhongmou County.
 In May 2019, the school visited the military museum of Henan Military
Region and take condolences to the soldiers.
 In June 2019, the school held a propaganda event of “Small Long
March” by singing red songs and showing love to the country.
Kindergarten in Luohe  The school organised the environmental protection initiative called
“Little Citizens, Big Voices” in Red Maple Plaza to promote the
environment protection concept through garbage collection and
environmental protection proposals.
Kindergarten in Xingyang  In March 2019, the school organised students to pick up garbage
in Tianjian Lake in Xingyang, aiming to strengthen the environment
protection awareness of students.
 In June 2019, the school held the activity of “Low-carbon
environment protection, love and charity sales, and make the best
use of objects”.
ENVIRONMENTAL, SOCIAL AND GOVERNANCE REPORT 2019 19
III. COMMUNITY INVESTMENT (CONTINUED)
Public Welfare Volunteer Activities
SOCIAL DONATION
The Group mainly focuses on social donations and sponsorship projects for local education, cultural
construction, poverty alleviation, assisting women and children, and environmental welfare activities.
Major projects include but are not limited to:
 The Group sponsored the Zhengzhou study and exchange activities of teachers and students in
Zhongyu Village Primary School of Luoyang City;
 The Group donated over RMB130 thousand to the local environment protection project “Otis
Tarda Return Home” in Henan;
 The Group donated over RMB22 thousand to the activity of “Aid with Love Activity of the Youth
League”;
 Kaifeng Yuhua High School donated over RMB70 thousand and office supplies to the poverty
alleviation by education; and
 Kaifeng Yuhua Elite School donated over 100 books to the Hand with Hand Primary School in
Lankao County.
20 CHINA YUHUA EDUCATION CORPORATION LIMITED
III. COMMUNITY INVESTMENT (CONTINUED)
STAFF CARE
As teaching staff are a significant asset of education providers, the Group has always shown great
care for our staff. After fully understanding the actual needs of our staff, we offer various support in the
form of money, materials, manpower, greetings and so on to help our staff overcome difficulties they
face with in daily life, in mental life and in their jobs.
The Group’s subordinate school, Hunan International Economics University, has carried out sending
warmth and assistant work for staff. The university has formed a fine tradition that the school must
visit staff under four scenarios, when someone is in hospital, losing loved ones, facing natural
calamities and man-made misfortunes. Meanwhile, the school continuously raised the level of
consolation fund, which was up to RMB500 thousand and RMB242 thousand was distributed to
staff have extremely poor problems. At the same time, the school union established a supporting
mechanism for the extremely poor teachers and staff, and established personal files for all staff with
problems. During the Reporting Period, the school applied for the provincial education foundation
“Love Candle” relief fund of RMB20,000 for two seriously ill teaching staff. The school also applied the
special poor assistance from the provincial education trade union for nearly 20 employees in difficulty.
In addition, primary and secondary schools and kindergartens in various districts have provided
assistance to assist staffs encountering difficulties in daily life and mental life with the actual needs of
staff, including funds, materials, human assistance and spiritual condolences. The Group also gives
benefits to staff during holidays such as the Women’s Day, the Dragon Boat Festival, the Mid-Autumn
Festival, Teacher’s Day, Spring Festival, as well as showing concerns to staff’s daily lives.
Over the past fiscal year, subordinate schools of the Group have devoted themselves in community
building and taken a good lead in their communities. During the Reporting Period, subordinate schools
of the Group have won over 30 awards.
ENVIRONMENTAL, SOCIAL AND GOVERNANCE REPORT 2019 21
IV. ENVIRONMENTAL
Strictly complying with applicable laws and regulations including the Environmental Protection Law,
the Atmospheric Pollution Prevention Law, the Water Pollution Prevention Law, the Solid Waste
Pollution Control Law, and the Energy Conservation Law in China, the Group ensures that the daily
operations of subordinate schools do not have a significant impact on the environment and natural
resources. The Group has formulated the Detailed Regulations for Energy Conservation Management
to effectively promote energy conservation and emission reduction in accordance with the spirit of the
Energy Conservation Management Measures of Henan Province. The regulation aims to minimise the
consumption of water, electricity and natural gas, and the emission of waste water and gas as much
as possible by management energy-saving, technical energy-saving and behavioural energy-saving,
realizing the effective and rational use of energy and promoting the construction of energy-saving
campus. During the Reporting Period, the Group did not violate relevant laws and regulations, and has
not received any complaints regarding the emission of waste gases, greenhouse gases and pollutants.
EMISSIONS
The Group is principally engaged in education services. No substantial emissions are produced by
combustion of any fuels in daily operation as the Group is not engaged in any industrial production.
During the Reporting Period, the principal type of emission of the Group is exhaust generated by the
Group’s self-owned vehicles. The main emission data are as follows:
Major emissions Unit
Emission
volume
Nitrogen oxide (NOx) Gram 673,012.0
Sulphur dioxide (SOx) Gram 637.2
Particulate Matter Gram 66,032.1
The Group does not generate any greenhouse gases through any fixed combustion source. The direct
emission of greenhouse gases is the exhaust produced by the Group’s self-owned vehicles. Indirect
greenhouse gas emission was mainly generated from the use of electricity, natural gas, etc. During the
Reporting Period, the Group’s emission type and data of major greenhouse gases are as below:
Greenhouse gases Unit
Emission
volume
Carbon dioxide Ton 49,100.4
Methane Ton 0.2
Nitrous oxide Ton 14.8
22 CHINA YUHUA EDUCATION CORPORATION LIMITED
IV. ENVIRONMENTAL (CONTINUED)
No hazardous waste is produced during daily operation of the Group. The non-hazardous wastes
generated by the Group mainly include garbage generated in the daily operation of schools such as
office supplies and food residues. After waste by the Group, waste is transferred by the municipal
disposal company (which satisfies legal and regulatory requirements) to the garbage transfer station
designated by environmental, health and other departments in line with relevant national and regional
treatment standards. Further, garbage collection areas of schools of the Group are disinfected at least
twice daily to ensure that it waste not substantially affect the school environment. In addition, the
wastewater pipelines of the Group’s campuses are handed over to professional dredging companies
for cleaning up to once a week. After initial sedimentation, the domestic wastewater generated during
the daily operation will be pumped into municipal wastewater pipeline and delivered to professional
wastewater treatment companies for further handling.
During the Reporting Period, the Group did not have specific statistical data on non-hazardous
wastes. The Group has always been focused on the classification, disposal and reuse of waste. While
implementing education and guidance work, the Group advocates the concept of “turning waste into
wealth and treasure”. Based on the Regulations on the Management of Waste Disposal by Yuhua
Education Group, the Group has established relevant waste recycling systems to further promote the
construction of the conservation and environmental-friendly school community.
The Group mainly reduces the emission of waste gases and greenhouse gases based on the
formulated policy of the Detailed Regulations for Energy Conservation Management. For instance,
promoting new type energy-saving and environmentally friendly products, encouraging the use of
water-saving and electricity-saving facilities and appliances, strictly controlling the procurement of
large-size equipment consuming large amount of energy, preferentially purchasing products that
meet national energy-saving standards with energy efficiency labels. In the meantime, the Group
continuously strengthens the recycling and discharge of the waste objects to prevent pollution
damage to the environment from the origin. During the Reporting Period, the emissions of the Group’s
major emission types have significantly declined compared with the previous fiscal year. Specifically,
nitrogen oxide declined by 17.6%, sulphur dioxide declined by 13.6% and particulate matter declined
by 17.7%. The Group’s subordinate schools have also achieved a reduction of approximately 500
tons in carbon dioxide by planting various types of trees.
USE OF RESOURCES
Promoting the good fashion of “saving being a glory, waste being a shame”, the Group regularly
conducts the emission reduction training according to the Training Program of Energy Conservation
and Emission Reduction, at the beginning of each year, and actively carries out energy-saving and
energy reduction actions by focusing on energy-saving, water-saving, electricity-saving and materialsaving engineer and build energy-saving schools.
ENVIRONMENTAL, SOCIAL AND GOVERNANCE REPORT 2019 23
IV. ENVIRONMENTAL (CONTINUED)
The Group has made rational use of energy and resources such as water, electricity and natural gas.
At the same time, we help students develop good habits for use of energy resources in order to limit
waste. During the Reporting Period, the main energy and resources consumption of the Group was as
follows:
Type of energy Unit Consumption
Total amount of electricity consumed Megawatt-hour 54,257.8
Intensity Megawatt-hour/
school
1,871.0
Total amount of water consumed Thousand ton 3,153.7
Intensity Thousand ton /school 108.7
Total natural gas consumption Thousand m3 925.0
Intensity Thousand m3
/school 31.9
Total gasoline consumption Litre 43,347.2
Intensity Litre/school 1,494.7
The Group’s energy use efficiency plan is mainly carried out in accordance with the Detailed
Regulations for Energy Conservation Management formulated by the Group, and enhances the staff’s
awareness of energy conservation and emission reduction by regularly training them in conjunction
with the Training Program of Energy Conservation and Emission Reduction at the beginning of each
year. The Group improves the utilisation of energy and resources by clarifying the responsibilities
of various departments within the school, and through measures such as scientific management,
technology upgrades, and behavioural training. For instance, we vigorously promote the application
of energy-efficient teaching equipment; we promote the construction of paperless office and
decrease the use of paper; we reform the water supply pipeline at campus and raise the repeated
use rate of water; we reconstruct the lighting facilities at campus and vigorously promote the energysaving lamps, as well as launching green lighting projects. The Group continuously strengthens the
management of the use of water, electricity and natural gas, and cultivates students’ awareness of
saving energy and water.
The Group has not encountered problems in sourcing water that is fit for purpose. All of the Group’s
schools have stable sources of water. In accordance with the Detailed Regulations for Energy
Conservation Management, the Group has formulated a strict water utilisation system through
management, supervision and charging method to save water. Further, the Group has implemented
reconstruction projects in the water supply pipeline at campus to raise the repeated use rate of water.
During the Reporting Period, the Group has achieved significant results in improving the efficiency of
using energy and resources. The total natural gas consumption has declined by 43.3%, and the total
gasoline consumption has declined by 13.6%.
As the Group does not manufacture any products, the Group does not use any kind of packaging
materials.
24 CHINA YUHUA EDUCATION CORPORATION LIMITED
IV. ENVIRONMENTAL (CONTINUED)
ENVIRONMENT AND NATURAL RESOURCES
The Group’s schools do not have any major impact on the environment and natural resources
during daily operation. The Group advocates “low-carbon traveling” to minimise the emission of
pollutants and greenhouse gases from vehicles. In the meantime, according to the formulated Detailed
Regulations for Energy Conservation Management, the Group constantly monitors the emission of
pollutants and the energy consumption of water, electricity and natural gas. The Group has formed
an energy management system centred with management energy saving, technical energy saving,
and behavioural energy saving to ensure that the company’s business activities have created the
lowest impact on the surrounding community environment and natural resources. To reduce the
potential threat to natural resources caused by possibly abusive use of paper, the Group formulated
the Provisions on the Use of Paper to reasonably regulate the use of office and teaching materials
as well as promoting a paperless office environment to minimise paper consumption at best efforts,
and ensuring the effective use of paper and eliminating paper waste. In addition, the Group strongly
promotes afforestation activities in order to protect the environment. During the Reporting Period, the
Group planted more than 200,000 trees and greened over 20 thousand square meters of lawn.
'''
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


example = {
    'text': [input],
    'path': ['emission'],
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
print(eval_set)


extractor = ESG(model_checkpoint)


eval_firuge_set = extractor.get_fiture_set(small_eval_set, eval_set)
print(eval_firuge_set)
