import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager, rc
import matplotlib.font_manager as fm
import folium
from geopy.geocoders import Nominatim
from folium import Map, Marker
import re



plt.matplotlib.use('Agg')

# Heroku 포트 설정 추가
port = os.environ.get("PORT", 8501)

st.title('지점데이터 분석')

# 탭 메뉴
tab1, tab2,tab3,tab4= st.tabs(["지점 분포현황","지점별 고객 거주지", "지점별 고객성별&연령","지점별 판매차종"])










# 사이드바에서 데이터 분류 선택
option = st.sidebar.radio("카테고리", ["아우디", "폭스바겐"], key=3)

if option == "아우디":

    with tab1:
        st.header("아우디 지점 분포현황")
        st.write("- 남천 1, 남천 2는 남천으로, 해운대1, 해운대2는 해운대로 통합 \n- 선등록, AAP, HQ, MAL 라고 되어있는 지점데이터는 제외")

        # 엑셀 파일 경로 설정
        file_path = 'data/audi_store.xlsx'
        # 엑셀 파일 읽기
        df = pd.read_excel(file_path)


        #font_path = os.path.join("data", "MALGUN.TTF")  # 데이터 폴더 내의 폰트 파일 경로

        font_path = 'fonts/MALGUN.TTF'  # 폰트 경로 설정
        font = font_manager.FontProperties(fname=font_path).get_name()
        rc('font', family=font)



        # 2. 지점명 통합 (남천1, 남천2는 남천으로 / 해운대1, 해운대2는 해운대로)
        df['지점명'] = df['지점명'].replace({
            '남천1': '남천', '남천2': '남천',
            '해운대1': '해운대', '해운대2': '해운대'
        })

        # 3. 제외할 지점 제거 (선등록, AAP, HQ, MAL 제거)
        df = df[~df['지점명'].isin(['선등록', 'AAP', 'HQ', 'MAL'])]

        # 4. 지점별로 개수 카운트
        branch_count = df['지점명'].value_counts()

        # 데이터프레임으로 변환
        branch_count_df = branch_count.reset_index()
        branch_count_df.columns = ['지점명', '고객 수']  # 열 이름 변경
        # 5. 전체 고객 수 계산
        total_customers = branch_count.sum()


        # 6. 결과 시각화 (막대 그래프) - 비율 표시 추가
        #st.subheader('지점별 분포현황')

        # 그래프 그리기
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=branch_count.index, y=branch_count.values, palette="Blues_d", ax=ax)

        # 막대 위에 비율 표시
        for index, value in enumerate(branch_count.values):
            percentage = (value / total_customers) * 100
            ax.text(index, value + 10, f"{percentage:.2f}%", ha='center')

        ax.set_title('지점별 분포현황')
        ax.set_xlabel('지점명')
        ax.set_ylabel('판매 개수')
        ax.set_xticks(range(len(branch_count)))
        ax.set_xticklabels(branch_count.index, rotation=45)

        # 그래프 Streamlit에 표시
        st.pyplot(fig)

        # Streamlit에서 데이터프레임 표시
        #st.subheader('지점별 판매 개수')

        st.dataframe(branch_count_df, use_container_width=True)


    with tab2:
    

        #st.header("<아우디>")

    
        #st.text("") 
        #st.header("지점 분포현황")
        #st.write("- 남천 1, 남천 2는 남천으로, 해운대1, 해운대2는 해운대로 통합 \n- 선등록, AAP, HQ, MAL 라고 되어있는 지점데이터는 제외")

        # 엑셀 파일 경로 설정
        file_path = 'data/audi_store.xlsx'
        # 엑셀 파일 읽기
        df = pd.read_excel(file_path)


        #font_path = os.path.join("data", "MALGUN.TTF")  # 데이터 폴더 내의 폰트 파일 경로

        font_path = 'fonts/MALGUN.TTF'  # 폰트 경로 설정
        font = font_manager.FontProperties(fname=font_path).get_name()
        rc('font', family=font)



        # 2. 지점명 통합 (남천1, 남천2는 남천으로 / 해운대1, 해운대2는 해운대로)
        df['지점명'] = df['지점명'].replace({
            '남천1': '남천', '남천2': '남천',
            '해운대1': '해운대', '해운대2': '해운대'
        })

        # 3. 제외할 지점 제거 (선등록, AAP, HQ, MAL 제거)
        df = df[~df['지점명'].isin(['선등록', 'AAP', 'HQ', 'MAL'])]

        # 4. 지점별로 개수 카운트
        branch_count = df['지점명'].value_counts()

        # 데이터프레임으로 변환
        branch_count_df = branch_count.reset_index()
        branch_count_df.columns = ['지점명', '고객 수']  # 열 이름 변경
        # 5. 전체 고객 수 계산
        total_customers = branch_count.sum()


        # 6. 결과 시각화 (막대 그래프) - 비율 표시 추가
        #st.subheader('지점별 분포현황')

        # 그래프 그리기
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=branch_count.index, y=branch_count.values, palette="Blues_d", ax=ax)

        # 막대 위에 비율 표시
        for index, value in enumerate(branch_count.values):
            percentage = (value / total_customers) * 100
            ax.text(index, value + 10, f"{percentage:.2f}%", ha='center')

        ax.set_title('지점별 분포현황')
        ax.set_xlabel('지점명')
        ax.set_ylabel('판매 개수')
        ax.set_xticks(range(len(branch_count)))
        ax.set_xticklabels(branch_count.index, rotation=45)

        # 그래프 Streamlit에 표시
        #st.pyplot(fig)

        # Streamlit에서 데이터프레임 표시
        #st.subheader('지점별 판매 개수')

        #st.dataframe(branch_count_df, use_container_width=True)

        #st.divider() 


        st.header("아우디 지점별 고객 거주지 분포현황")




        # 지점 선택을 위한 드롭다운 메뉴
        selected_branch = st.selectbox('지점을 선택하세요:', df['지점명'].unique())

        # 선택된 지점에 해당하는 데이터 필터링
        selected_df = df[df['지점명'] == selected_branch]

        # 주소에서 첫 번째 부분(도시 또는 시, 구 등) 추출
        selected_df['거주지'] = selected_df['주소'].apply(lambda x: str(x).split()[0])

        # 거주지별로 카운트
        address_count = selected_df['거주지'].value_counts()

        # 선택된 지점 거주지 분포 데이터 정리
        address_count_df = pd.DataFrame(address_count).reset_index()
        address_count_df.columns = ['거주지', '카운트']

        # 전체 고객 수 계산
        total_customers = address_count_df['카운트'].sum()

        # 각 거주지별 비율 계산
        address_count_df['비율 (%)'] = (address_count_df['카운트'] / total_customers) * 100


        # 시각화 (막대 그래프) - 비율 표시 추가
        st.subheader(f'{selected_branch} 지점 고객 거주지 분포 및 비율')

        # 그래프 그리기
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(x=address_count_df['거주지'], y=address_count_df['카운트'], palette="Blues_d", ax=ax)

        # 막대 위에 비율 표시
        for index, value in enumerate(address_count_df['카운트']):
            ax.text(index, value + 10, f"{address_count_df['비율 (%)'].iloc[index]:.2f}%", ha='center')

        ax.set_title(f'{selected_branch} 지점 고객 거주지 분포 및 비율')
        ax.set_xlabel('거주지')
        ax.set_ylabel('고객 수')
        ax.set_xticks(range(len(address_count_df)))
        ax.set_xticklabels(address_count_df['거주지'], rotation=45)

        # 그래프를 Streamlit에 표시
        st.pyplot(fig)

        # Streamlit에서 데이터프레임 표시
        st.dataframe(address_count_df, use_container_width=True)

        st.divider() 


        if(selected_branch=="남천" or selected_branch=="해운대"):
            st.subheader(f'{selected_branch} 지점 부산 고객 거주지 분포 및 비율')

            # 주소에서 첫 번째와 두 번째 부분(도시 또는 시, 구 등) 추출
            #selected_df['거주지2'] = selected_df['주소'].apply(lambda x: ' '.join(str(x).split()[:2]))

            # '부산'인 값에 대해서만 필터링하여 새로운 데이터프레임 생성
            busan_df = selected_df[selected_df['거주지'] == '부산'].copy()

            busan_df['주소'] = busan_df['주소'].apply(lambda x: ' '.join(str(x).split()[:2]))

            # 결과 확인 (이 부분은 필요에 따라 추가할 수 있습니다)
            #st.dataframe(busan_df)


            # 거주지별로 카운트
            address_count = busan_df['주소'].value_counts()

            # 선택된 지점 거주지 분포 데이터 정리
            address_count_df = pd.DataFrame(address_count).reset_index()
            address_count_df.columns = ['거주지', '카운트']

            # 전체 고객 수 계산
            total_customers = address_count_df['카운트'].sum()

            # 각 거주지별 비율 계산
            address_count_df['비율 (%)'] = (address_count_df['카운트'] / total_customers) * 100

            
            # 그래프 그리기
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(x=address_count_df['거주지'], y=address_count_df['카운트'], palette="Blues_d", ax=ax)

            # 막대 위에 비율 표시
            for index, value in enumerate(address_count_df['카운트']):
                ax.text(index, value + 10, f"{address_count_df['비율 (%)'].iloc[index]:.2f}%", ha='center')

            ax.set_title(f'{selected_branch} 지점 부산 고객 거주지 분포 및 비율')
            ax.set_xlabel('거주지')
            ax.set_ylabel('고객 수')
            ax.set_xticks(range(len(address_count_df)))
            ax.set_xticklabels(address_count_df['거주지'], rotation=45)

            # 그래프를 Streamlit에 표시
            st.pyplot(fig)

            st.dataframe(address_count_df, use_container_width=True)



        elif(selected_branch=="창원"):

            st.subheader(f'{selected_branch} 지점 경남 고객 거주지 분포 및 비율')

            # 주소에서 첫 번째와 두 번째 부분(도시 또는 시, 구 등) 추출
            #selected_df['거주지2'] = selected_df['주소'].apply(lambda x: ' '.join(str(x).split()[:2]))

            # '부산'인 값에 대해서만 필터링하여 새로운 데이터프레임 생성
            busan_df = selected_df[selected_df['거주지'] == '경남'].copy()

            busan_df['주소'] = busan_df['주소'].apply(lambda x: ' '.join(str(x).split()[:2]))

            # 결과 확인 (이 부분은 필요에 따라 추가할 수 있습니다)
            #st.dataframe(busan_df)


            # 거주지별로 카운트
            address_count = busan_df['주소'].value_counts()

            # 선택된 지점 거주지 분포 데이터 정리
            address_count_df = pd.DataFrame(address_count).reset_index()
            address_count_df.columns = ['거주지', '카운트']

            # 전체 고객 수 계산
            total_customers = address_count_df['카운트'].sum()

            # 각 거주지별 비율 계산
            address_count_df['비율 (%)'] = (address_count_df['카운트'] / total_customers) * 100

            
            # 그래프 그리기
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(x=address_count_df['거주지'], y=address_count_df['카운트'], palette="Blues_d", ax=ax)

            # 막대 위에 비율 표시
            for index, value in enumerate(address_count_df['카운트']):
                ax.text(index, value + 10, f"{address_count_df['비율 (%)'].iloc[index]:.2f}%", ha='center')

            ax.set_title(f'{selected_branch} 지점 부산 고객 거주지 분포 및 비율')
            ax.set_xlabel('거주지')
            ax.set_ylabel('고객 수')
            ax.set_xticks(range(len(address_count_df)))
            ax.set_xticklabels(address_count_df['거주지'], rotation=45)

            # 그래프를 Streamlit에 표시
            st.pyplot(fig)

            st.dataframe(address_count_df, use_container_width=True)


        elif(selected_branch=="울산"):

            st.subheader(f'{selected_branch} 지점 경남 고객 거주지 분포 및 비율')

            # 주소에서 첫 번째와 두 번째 부분(도시 또는 시, 구 등) 추출
            #selected_df['거주지2'] = selected_df['주소'].apply(lambda x: ' '.join(str(x).split()[:2]))

            # '부산'인 값에 대해서만 필터링하여 새로운 데이터프레임 생성
            busan_df = selected_df[selected_df['거주지'] == '울산'].copy()

            busan_df['주소'] = busan_df['주소'].apply(lambda x: ' '.join(str(x).split()[:2]))

            # 결과 확인 (이 부분은 필요에 따라 추가할 수 있습니다)
            #st.dataframe(busan_df)


            # 거주지별로 카운트
            address_count = busan_df['주소'].value_counts()

            # 선택된 지점 거주지 분포 데이터 정리
            address_count_df = pd.DataFrame(address_count).reset_index()
            address_count_df.columns = ['거주지', '카운트']

            # 전체 고객 수 계산
            total_customers = address_count_df['카운트'].sum()

            # 각 거주지별 비율 계산
            address_count_df['비율 (%)'] = (address_count_df['카운트'] / total_customers) * 100

            
            # 그래프 그리기
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(x=address_count_df['거주지'], y=address_count_df['카운트'], palette="Blues_d", ax=ax)

            # 막대 위에 비율 표시
            for index, value in enumerate(address_count_df['카운트']):
                ax.text(index, value + 10, f"{address_count_df['비율 (%)'].iloc[index]:.2f}%", ha='center')

            ax.set_title(f'{selected_branch} 지점 부산 고객 거주지 분포 및 비율')
            ax.set_xlabel('거주지')
            ax.set_ylabel('고객 수')
            ax.set_xticks(range(len(address_count_df)))
            ax.set_xticklabels(address_count_df['거주지'], rotation=45)

            # 그래프를 Streamlit에 표시
            st.pyplot(fig)

            st.dataframe(address_count_df, use_container_width=True)
        
        
        elif(selected_branch=="진주"):

            st.subheader(f'{selected_branch} 지점 경남 고객 거주지 분포 및 비율')

            # 주소에서 첫 번째와 두 번째 부분(도시 또는 시, 구 등) 추출
            #selected_df['거주지2'] = selected_df['주소'].apply(lambda x: ' '.join(str(x).split()[:2]))

            # '부산'인 값에 대해서만 필터링하여 새로운 데이터프레임 생성
            busan_df = selected_df[selected_df['거주지'] == '경남'].copy()

            busan_df['주소'] = busan_df['주소'].apply(lambda x: ' '.join(str(x).split()[:2]))

            # 결과 확인 (이 부분은 필요에 따라 추가할 수 있습니다)
            #st.dataframe(busan_df)


            # 거주지별로 카운트
            address_count = busan_df['주소'].value_counts()

            # 선택된 지점 거주지 분포 데이터 정리
            address_count_df = pd.DataFrame(address_count).reset_index()
            address_count_df.columns = ['거주지', '카운트']

            # 전체 고객 수 계산
            total_customers = address_count_df['카운트'].sum()

            # 각 거주지별 비율 계산
            address_count_df['비율 (%)'] = (address_count_df['카운트'] / total_customers) * 100

            
            # 그래프 그리기
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(x=address_count_df['거주지'], y=address_count_df['카운트'], palette="Blues_d", ax=ax)

            # 막대 위에 비율 표시
            for index, value in enumerate(address_count_df['카운트']):
                ax.text(index, value + 10, f"{address_count_df['비율 (%)'].iloc[index]:.2f}%", ha='center')

            ax.set_title(f'{selected_branch} 지점 부산 고객 거주지 분포 및 비율')
            ax.set_xlabel('거주지')
            ax.set_ylabel('고객 수')
            ax.set_xticks(range(len(address_count_df)))
            ax.set_xticklabels(address_count_df['거주지'], rotation=45)

            # 그래프를 Streamlit에 표시
            st.pyplot(fig)

            st.dataframe(address_count_df, use_container_width=True)


        
        elif(selected_branch=="제주"):

            st.subheader(f'{selected_branch} 지점 시별 거주지 분포 및 비율')

            # 주소에서 첫 번째와 두 번째 부분(도시 또는 시, 구 등) 추출
            #selected_df['거주지2'] = selected_df['주소'].apply(lambda x: ' '.join(str(x).split()[:2]))

            # '부산'인 값에 대해서만 필터링하여 새로운 데이터프레임 생성
            busan_df = selected_df[selected_df['거주지'] == '제주특별자치도'].copy()

            busan_df['주소'] = busan_df['주소'].apply(lambda x: ' '.join(str(x).split()[:2]))

            # 결과 확인 (이 부분은 필요에 따라 추가할 수 있습니다)
            #st.dataframe(busan_df)


            # 거주지별로 카운트
            address_count = busan_df['주소'].value_counts()

            # 선택된 지점 거주지 분포 데이터 정리
            address_count_df = pd.DataFrame(address_count).reset_index()
            address_count_df.columns = ['거주지', '카운트']

            # 전체 고객 수 계산
            total_customers = address_count_df['카운트'].sum()

            # 각 거주지별 비율 계산
            address_count_df['비율 (%)'] = (address_count_df['카운트'] / total_customers) * 100

            
            # 그래프 그리기
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(x=address_count_df['거주지'], y=address_count_df['카운트'], palette="Blues_d", ax=ax)

            # 막대 위에 비율 표시
            for index, value in enumerate(address_count_df['카운트']):
                ax.text(index, value + 10, f"{address_count_df['비율 (%)'].iloc[index]:.2f}%", ha='center')

            ax.set_title(f'{selected_branch} 지점 부산 고객 거주지 분포 및 비율')
            ax.set_xlabel('거주지')
            ax.set_ylabel('고객 수')
            ax.set_xticks(range(len(address_count_df)))
            ax.set_xticklabels(address_count_df['거주지'], rotation=45)

            # 그래프를 Streamlit에 표시
            st.pyplot(fig)

            st.dataframe(address_count_df, use_container_width=True)

    ###################

            
        with tab3:
            st.header("아우디 고객성별&연령")
            #st.write("여기에 아우디 페이지 내용을 추가하세요.")

        # 성별과 연령을 분리
            df[['성별', '연령']] = df['성별/연령'].str.split('/', expand=True)

            # 성별 데이터를 확인하기 위한 고유 값 출력
            #st.header("고객 성별 데이터 종류")
            #unique_genders = df['성별'].unique()
            #st.write(f"성별 데이터에 포함된 값들: {unique_genders}")

            # 성별 데이터를 통일
            df['성별'] = df['성별'].str.strip()  # 공백 제거
            df['성별'] = df['성별'].replace({
                '남자': '남성',
                '남': '남성',
                '남성': '남성',
                '여자': '여성',
                '여': '여성',
                '여성': '여성'
            })

            # 불필요한 값들을 제거 (유효한 값은 '남성'과 '여성'만)
            df = df[df['성별'].isin(['남성', '여성'])]
                        
            # 연령을 숫자로 변환
            df['연령'] = pd.to_numeric(df['연령'], errors='coerce')
            
            # 지점 선택을 위한 드롭다운 메뉴 (고유한 key 부여)
            selected_branch3 = st.selectbox('지점을 선택하세요:', df['지점명'].unique(), key="selectbox3")
            
            # 선택된 지점에 해당하는 데이터 필터링
            selected_df = df[df['지점명'] == selected_branch3]
            
            # 성별 및 연령대별로 카운트
            gender_count = selected_df['성별'].value_counts(normalize=True) * 100  # 성별 비율
            age_group_count = selected_df['연령'].value_counts(bins=[0, 20, 30, 40, 50,  60, 100], normalize=True) * 100  # 연령대 비율

            # 성별 비율 시각화
            st.subheader(f'{selected_branch3} 지점 고객 성별 비율')
            fig_gender, ax_gender = plt.subplots(figsize=(6, 4))
            sns.barplot(x=gender_count.index, y=gender_count.values, palette="Blues_d", ax=ax_gender)

            # 막대 위에 비율 텍스트 추가
            for i, value in enumerate(gender_count.values):
                ax_gender.text(i, value + 1, f'{value:.2f}%', ha='center')


            ax_gender.set_xlabel('성별')
            ax_gender.set_ylabel('비율 (%)')
            ax_gender.set_title(f'{selected_branch3} 지점 고객 성별 비율')
            st.pyplot(fig_gender)

            gender_df = pd.DataFrame({
                '성별': gender_count.index,
                '비율': [f'{value:.2f}%' for value in gender_count.values]
            })

            # 성별 비율 테이블 출력
            st.table(gender_df)

        


            # 연령대 구간 라벨 설정
            age_group_labels = ['0~20세', '20대', '30대', '40대', '50대', '60세 이상']

            # x축 라벨을 맞추기 위해 인덱스를 순서대로 설정
            age_group_count.index = pd.IntervalIndex(age_group_count.index)
            age_group_count = age_group_count.sort_index()  # 연령대를 어린 나이순으로 정렬

            # 연령대 비율 시각화
            st.subheader(f'{selected_branch3} 지점 고객 연령대 비율')
            fig_age, ax_age = plt.subplots(figsize=(6, 4))

            # 막대 그래프 그리기
            sns.barplot(x=age_group_labels, y=age_group_count.values, palette="Blues_d", ax=ax_age)

            # 막대 위에 비율 텍스트 추가
            for i, value in enumerate(age_group_count.values):
                ax_age.text(i, value + 1, f'{value:.2f}%', ha='center')

            ax_age.set_xlabel('연령대')
            ax_age.set_ylabel('비율 (%)')
            ax_age.set_title(f'{selected_branch3} 지점 고객 연령대 비율')

            # 그래프를 Streamlit에 표시
            st.pyplot(fig_age)

            # 연령대 비율 테이블 출력
            age_df = pd.DataFrame({
                '연령대': age_group_labels,
                '비율': [f'{value:.2f}%' for value in age_group_count.values]
            })
            st.table(age_df)


        with tab4:
            st.header("아우디 지점별 판매차종")

            # 모델명에서 괄호와 그 안의 내용을 제거하는 함수
            def clean_model(model):
                model_without_brackets = re.sub(r'\(.*?\)', '', model).strip()
                return model_without_brackets.split()[0].strip()

            # 모델 데이터를 가공하여 괄호 내용 제거
            df['모델'] = df['모델'].apply(clean_model)

            # 전체 지점 판매대수 보기 버튼을 상단에 배치
            st.write("") 
            if st.button('전체 지점 모델 판매대수 보기', key="all_branches"):
                model_count_total = df['모델'].value_counts()  # 전체 지점의 모델별 판매대수
                total_sales = model_count_total.sum()  # 전체 판매대수
                model_percentage_total = (model_count_total / total_sales) * 100  # 전체 지점의 모델별 판매 비율

                st.subheader('전체 지점 모델별 판매 비율')
                fig_total_model, ax_total_model = plt.subplots(figsize=(6, 4))
                sns.barplot(x=model_count_total.index, y=model_percentage_total.values, palette="Blues_d", ax=ax_total_model)

                # 막대 위에 비율 텍스트 추가
                for i, value in enumerate(model_percentage_total.values):
                    ax_total_model.text(i, value + 1, f'{value:.2f}%', ha='center')

                ax_total_model.set_xlabel('모델')
                ax_total_model.set_ylabel('판매 비율 (%)')
                ax_total_model.set_title('전체 지점 모델별 판매 비율')
                st.pyplot(fig_total_model)

                # 전체 지점 모델별 판매대수 및 비율 테이블 출력
                model_df_total = pd.DataFrame({
                    '모델': model_count_total.index,
                    '판매대수': [f'{int(value)}대' for value in model_count_total.values],
                    '비율 (%)': [f'{value:.2f}%' for value in model_percentage_total.values]
                })

                st.dataframe(model_df_total, use_container_width=True)

            st.write("")  # 공간 추가

            # 지점 선택을 위한 드롭다운 메뉴 (고유한 key 부여)
            selected_branch4 = st.selectbox('지점을 선택하세요:', df['지점명'].unique(), key="selectbox5")

            # 선택한 지점에 해당하는 데이터 필터링
            filtered_df = df[df['지점명'] == selected_branch4]

            # 모델별 판매대수 및 비율 계산
            model_count_filtered = filtered_df['모델'].value_counts()
            total_sales_filtered = model_count_filtered.sum()
            model_percentage_filtered = (model_count_filtered / total_sales_filtered) * 100

            # 모델별 판매대수 시각화 (비율로 변경)
            st.subheader(f'{selected_branch4} 지점 모델별 판매 비율')
            fig_model, ax_model = plt.subplots(figsize=(6, 4))
            sns.barplot(x=model_count_filtered.index, y=model_percentage_filtered.values, palette="Blues_d", ax=ax_model)

            # 막대 위에 비율 텍스트 추가
            for i, value in enumerate(model_percentage_filtered.values):
                ax_model.text(i, value + 1, f'{value:.2f}%', ha='center')

            ax_model.set_xlabel('모델')
            ax_model.set_ylabel('판매 비율 (%)')
            ax_model.set_title(f'{selected_branch4} 지점 모델별 판매 비율')
            st.pyplot(fig_model)

            # 모델별 판매대수 및 비율 테이블 출력
            model_df_filtered = pd.DataFrame({
                '모델': model_count_filtered.index,
                '판매대수': [f'{int(value)}대' for value in model_count_filtered.values],
                '비율 (%)': [f'{value:.2f}%' for value in model_percentage_filtered.values]
            })

            # 모델별 판매대수 데이터프레임 출력
            st.dataframe(model_df_filtered, use_container_width=True)

            st.write("")  # 공간 추가     










elif option == "폭스바겐":

    with tab1:
        st.header("폭스바겐 지점 분포현황")
        st.write("- 창원1, 창원2 지점 창원으로 통합\n - 선등록, AAP, HQ, MAL, VMA 제외")

        # 엑셀 파일 경로 설정
        file_path = 'data/vx_store.xlsx'
        # 엑셀 파일 읽기
        df = pd.read_excel(file_path)

        #font_path = os.path.join("data", "MALGUN.TTF")

        font_path = 'fonts/MALGUN.TTF'  # 폰트 경로 설정
        font_prop = fm.FontProperties(fname=font_path)
        # 전역 폰트 설정
        plt.rcParams['font.family'] = font_prop.get_name()
        #font = font_manager.FontProperties(fname=font_path).get_name()
        #rc('font', family=font)


        df['지점명'] = df['지점명'].replace({
        '창원1': '창원', '창원2': '창원'
        })

        # 3. 제외할 지점 제거 (선등록, AAP, HQ, MAL 제거)
        df = df[~df['지점명'].isin(['선등록', 'AAP', 'HQ', 'MAL','VWA'])]

        # 4. 지점별로 개수 카운트
        branch_count = df['지점명'].value_counts()

        # 데이터프레임으로 변환
        branch_count_df = branch_count.reset_index()
        branch_count_df.columns = ['지점명', '고객 수']  # 열 이름 변경
        # 5. 전체 고객 수 계산
        total_customers = branch_count.sum()


        # 6. 결과 시각화 (막대 그래프) - 비율 표시 추가
        #st.subheader('지점별 분포현황')

        # 그래프 그리기
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=branch_count.index, y=branch_count.values, palette="Blues_d", ax=ax)

        # 막대 위에 비율 표시
        for index, value in enumerate(branch_count.values):
            percentage = (value / total_customers) * 100
            ax.text(index, value + 10, f"{percentage:.2f}%", ha='center')

        plt.title('지점별 분포현황')
        plt.xlabel('지점명')
        plt.ylabel('판매 개수')
        ax.set_xticks(range(len(branch_count)))
        ax.set_xticklabels(branch_count.index, rotation=45)

        # 그래프 Streamlit에 표시
        st.pyplot(fig)

        # Streamlit에서 데이터프레임 표시
        #st.subheader('지점별 판매 개수')

        st.dataframe(branch_count_df, use_container_width=True)

        #st.divider() 



    with tab2:

        #st.text("") 
        #st.header("지점 분포현황")
        #st.write("- 창원1, 창원2 지점 창원으로 통합\n - 선등록, AAP, HQ, MAL, VMA 제외")

        # 엑셀 파일 경로 설정
        file_path = 'data/vx_store.xlsx'
        # 엑셀 파일 읽기
        df = pd.read_excel(file_path)

        #font_path = os.path.join("data", "MALGUN.TTF")

        font_path = 'fonts/MALGUN.TTF'  # 폰트 경로 설정
        font_prop = fm.FontProperties(fname=font_path)
        # 전역 폰트 설정
        plt.rcParams['font.family'] = font_prop.get_name()
        #font = font_manager.FontProperties(fname=font_path).get_name()
        #rc('font', family=font)


        df['지점명'] = df['지점명'].replace({
        '창원1': '창원', '창원2': '창원'
        })

        # 3. 제외할 지점 제거 (선등록, AAP, HQ, MAL 제거)
        df = df[~df['지점명'].isin(['선등록', 'AAP', 'HQ', 'MAL','VWA'])]

        # 4. 지점별로 개수 카운트
        branch_count = df['지점명'].value_counts()

        # 데이터프레임으로 변환
        branch_count_df = branch_count.reset_index()
        branch_count_df.columns = ['지점명', '고객 수']  # 열 이름 변경
        # 5. 전체 고객 수 계산
        total_customers = branch_count.sum()


        # 6. 결과 시각화 (막대 그래프) - 비율 표시 추가
        #st.subheader('지점별 분포현황')

        # 그래프 그리기
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=branch_count.index, y=branch_count.values, palette="Blues_d", ax=ax)

        # 막대 위에 비율 표시
        for index, value in enumerate(branch_count.values):
            percentage = (value / total_customers) * 100
            ax.text(index, value + 10, f"{percentage:.2f}%", ha='center')

        plt.title('지점별 분포현황')
        plt.xlabel('지점명')
        plt.ylabel('판매 개수')
        ax.set_xticks(range(len(branch_count)))
        ax.set_xticklabels(branch_count.index, rotation=45)

        # 그래프 Streamlit에 표시
        #st.pyplot(fig)

        # Streamlit에서 데이터프레임 표시
        #st.subheader('지점별 판매 개수')

        #st.dataframe(branch_count_df, use_container_width=True)

        #st.divider() 


        st.header("폭스바겐 지점별 고객 거주지 분포현황")




        # 지점 선택을 위한 드롭다운 메뉴
        selected_branch = st.selectbox('지점을 선택하세요:', df['지점명'].unique())

        # 선택된 지점에 해당하는 데이터 필터링
        selected_df = df[df['지점명'] == selected_branch]

        # 주소에서 첫 번째 부분(도시 또는 시, 구 등) 추출
        selected_df['거주지'] = selected_df['주소'].apply(lambda x: str(x).split()[0])

        # 거주지별로 카운트
        address_count = selected_df['거주지'].value_counts()

        # 선택된 지점 거주지 분포 데이터 정리
        address_count_df = pd.DataFrame(address_count).reset_index()
        address_count_df.columns = ['거주지', '카운트']

        # 전체 고객 수 계산
        total_customers = address_count_df['카운트'].sum()

        # 각 거주지별 비율 계산
        address_count_df['비율 (%)'] = (address_count_df['카운트'] / total_customers) * 100


        # 시각화 (막대 그래프) - 비율 표시 추가
        st.subheader(f'{selected_branch} 지점 고객 거주지 분포 및 비율')

        # 그래프 그리기
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(x=address_count_df['거주지'], y=address_count_df['카운트'], palette="Blues_d", ax=ax)

        # 막대 위에 비율 표시
        for index, value in enumerate(address_count_df['카운트']):
            ax.text(index, value + 10, f"{address_count_df['비율 (%)'].iloc[index]:.2f}%", ha='center')

        plt.title(f'{selected_branch} 지점 고객 거주지 분포 및 비율')
        plt.xlabel('거주지')
        plt.ylabel('고객 수')
        ax.set_xticks(range(len(address_count_df)))
        ax.set_xticklabels(address_count_df['거주지'], rotation=45)

        # 그래프를 Streamlit에 표시
        st.pyplot(fig)

        # Streamlit에서 데이터프레임 표시
        st.dataframe(address_count_df, use_container_width=True)


        if(selected_branch=="남천" or selected_branch=="해운대" or selected_branch=="동래"):
            st.subheader(f'{selected_branch} 지점 부산 고객 거주지 분포 및 비율')

            # 주소에서 첫 번째와 두 번째 부분(도시 또는 시, 구 등) 추출
            #selected_df['거주지2'] = selected_df['주소'].apply(lambda x: ' '.join(str(x).split()[:2]))

            # '부산'인 값에 대해서만 필터링하여 새로운 데이터프레임 생성
            busan_df = selected_df[selected_df['거주지'] == '부산'].copy()

            busan_df['주소'] = busan_df['주소'].apply(lambda x: ' '.join(str(x).split()[:2]))

            # 결과 확인 (이 부분은 필요에 따라 추가할 수 있습니다)
            #st.dataframe(busan_df)


            # 거주지별로 카운트
            address_count = busan_df['주소'].value_counts()

            # 선택된 지점 거주지 분포 데이터 정리
            address_count_df = pd.DataFrame(address_count).reset_index()
            address_count_df.columns = ['거주지', '카운트']

            # 전체 고객 수 계산
            total_customers = address_count_df['카운트'].sum()

            # 각 거주지별 비율 계산
            address_count_df['비율 (%)'] = (address_count_df['카운트'] / total_customers) * 100

            
            # 그래프 그리기
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(x=address_count_df['거주지'], y=address_count_df['카운트'], palette="Blues_d", ax=ax)

            # 막대 위에 비율 표시
            for index, value in enumerate(address_count_df['카운트']):
                ax.text(index, value + 10, f"{address_count_df['비율 (%)'].iloc[index]:.2f}%", ha='center')

            ax.set_title(f'{selected_branch} 지점 부산 고객 거주지 분포 및 비율')
            ax.set_xlabel('거주지')
            ax.set_ylabel('고객 수')
            ax.set_xticks(range(len(address_count_df)))
            ax.set_xticklabels(address_count_df['거주지'], rotation=45)

            # 그래프를 Streamlit에 표시
            st.pyplot(fig)

            st.dataframe(address_count_df, use_container_width=True)



        elif(selected_branch=="창원"):

            st.subheader(f'{selected_branch} 지점 경남 고객 거주지 분포 및 비율')

            # 주소에서 첫 번째와 두 번째 부분(도시 또는 시, 구 등) 추출
            #selected_df['거주지2'] = selected_df['주소'].apply(lambda x: ' '.join(str(x).split()[:2]))

            # '부산'인 값에 대해서만 필터링하여 새로운 데이터프레임 생성
            busan_df = selected_df[selected_df['거주지'] == '경남'].copy()

            busan_df['주소'] = busan_df['주소'].apply(lambda x: ' '.join(str(x).split()[:2]))

            # 결과 확인 (이 부분은 필요에 따라 추가할 수 있습니다)
            #st.dataframe(busan_df)


            # 거주지별로 카운트
            address_count = busan_df['주소'].value_counts()

            # 선택된 지점 거주지 분포 데이터 정리
            address_count_df = pd.DataFrame(address_count).reset_index()
            address_count_df.columns = ['거주지', '카운트']

            # 전체 고객 수 계산
            total_customers = address_count_df['카운트'].sum()

            # 각 거주지별 비율 계산
            address_count_df['비율 (%)'] = (address_count_df['카운트'] / total_customers) * 100

            
            # 그래프 그리기
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(x=address_count_df['거주지'], y=address_count_df['카운트'], palette="Blues_d", ax=ax)

            # 막대 위에 비율 표시
            for index, value in enumerate(address_count_df['카운트']):
                ax.text(index, value + 10, f"{address_count_df['비율 (%)'].iloc[index]:.2f}%", ha='center')

            ax.set_title(f'{selected_branch} 지점 부산 고객 거주지 분포 및 비율')
            ax.set_xlabel('거주지')
            ax.set_ylabel('고객 수')
            ax.set_xticks(range(len(address_count_df)))
            ax.set_xticklabels(address_count_df['거주지'], rotation=45)

            # 그래프를 Streamlit에 표시
            st.pyplot(fig)

            st.dataframe(address_count_df, use_container_width=True)


        elif(selected_branch=="울산"):

            st.subheader(f'{selected_branch} 지점 경남 고객 거주지 분포 및 비율')

            # 주소에서 첫 번째와 두 번째 부분(도시 또는 시, 구 등) 추출
            #selected_df['거주지2'] = selected_df['주소'].apply(lambda x: ' '.join(str(x).split()[:2]))

            # '부산'인 값에 대해서만 필터링하여 새로운 데이터프레임 생성
            busan_df = selected_df[selected_df['거주지'] == '울산'].copy()

            busan_df['주소'] = busan_df['주소'].apply(lambda x: ' '.join(str(x).split()[:2]))

            # 결과 확인 (이 부분은 필요에 따라 추가할 수 있습니다)
            #st.dataframe(busan_df)


            # 거주지별로 카운트
            address_count = busan_df['주소'].value_counts()

            # 선택된 지점 거주지 분포 데이터 정리
            address_count_df = pd.DataFrame(address_count).reset_index()
            address_count_df.columns = ['거주지', '카운트']

            # 전체 고객 수 계산
            total_customers = address_count_df['카운트'].sum()

            # 각 거주지별 비율 계산
            address_count_df['비율 (%)'] = (address_count_df['카운트'] / total_customers) * 100

            
            # 그래프 그리기
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(x=address_count_df['거주지'], y=address_count_df['카운트'], palette="Blues_d", ax=ax)

            # 막대 위에 비율 표시
            for index, value in enumerate(address_count_df['카운트']):
                ax.text(index, value + 10, f"{address_count_df['비율 (%)'].iloc[index]:.2f}%", ha='center')

            ax.set_title(f'{selected_branch} 지점 부산 고객 거주지 분포 및 비율')
            ax.set_xlabel('거주지')
            ax.set_ylabel('고객 수')
            ax.set_xticks(range(len(address_count_df)))
            ax.set_xticklabels(address_count_df['거주지'], rotation=45)

            # 그래프를 Streamlit에 표시
            st.pyplot(fig)

            st.dataframe(address_count_df, use_container_width=True)
        
        
        elif(selected_branch=="진주"):

            st.subheader(f'{selected_branch} 지점 경남 고객 거주지 분포 및 비율')

            # 주소에서 첫 번째와 두 번째 부분(도시 또는 시, 구 등) 추출
            #selected_df['거주지2'] = selected_df['주소'].apply(lambda x: ' '.join(str(x).split()[:2]))

            # '부산'인 값에 대해서만 필터링하여 새로운 데이터프레임 생성
            busan_df = selected_df[selected_df['거주지'] == '경남'].copy()

            busan_df['주소'] = busan_df['주소'].apply(lambda x: ' '.join(str(x).split()[:2]))

            # 결과 확인 (이 부분은 필요에 따라 추가할 수 있습니다)
            #st.dataframe(busan_df)


            # 거주지별로 카운트
            address_count = busan_df['주소'].value_counts()

            # 선택된 지점 거주지 분포 데이터 정리
            address_count_df = pd.DataFrame(address_count).reset_index()
            address_count_df.columns = ['거주지', '카운트']

            # 전체 고객 수 계산
            total_customers = address_count_df['카운트'].sum()

            # 각 거주지별 비율 계산
            address_count_df['비율 (%)'] = (address_count_df['카운트'] / total_customers) * 100

            
            # 그래프 그리기
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(x=address_count_df['거주지'], y=address_count_df['카운트'], palette="Blues_d", ax=ax)

            # 막대 위에 비율 표시
            for index, value in enumerate(address_count_df['카운트']):
                ax.text(index, value + 10, f"{address_count_df['비율 (%)'].iloc[index]:.2f}%", ha='center')

            ax.set_title(f'{selected_branch} 지점 부산 고객 거주지 분포 및 비율')
            ax.set_xlabel('거주지')
            ax.set_ylabel('고객 수')
            ax.set_xticks(range(len(address_count_df)))
            ax.set_xticklabels(address_count_df['거주지'], rotation=45)

            # 그래프를 Streamlit에 표시
            st.pyplot(fig)

            st.dataframe(address_count_df, use_container_width=True)


        
        elif(selected_branch=="제주"):

            st.subheader(f'{selected_branch} 지점 시별 거주지 분포 및 비율')

            # 주소에서 첫 번째와 두 번째 부분(도시 또는 시, 구 등) 추출
            #selected_df['거주지2'] = selected_df['주소'].apply(lambda x: ' '.join(str(x).split()[:2]))

            # '부산'인 값에 대해서만 필터링하여 새로운 데이터프레임 생성
            busan_df = selected_df[selected_df['거주지'] == '제주특별자치도'].copy()

            busan_df['주소'] = busan_df['주소'].apply(lambda x: ' '.join(str(x).split()[:2]))

            # 결과 확인 (이 부분은 필요에 따라 추가할 수 있습니다)
            #st.dataframe(busan_df)


            # 거주지별로 카운트
            address_count = busan_df['주소'].value_counts()

            # 선택된 지점 거주지 분포 데이터 정리
            address_count_df = pd.DataFrame(address_count).reset_index()
            address_count_df.columns = ['거주지', '카운트']

            # 전체 고객 수 계산
            total_customers = address_count_df['카운트'].sum()

            # 각 거주지별 비율 계산
            address_count_df['비율 (%)'] = (address_count_df['카운트'] / total_customers) * 100

            
            # 그래프 그리기
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(x=address_count_df['거주지'], y=address_count_df['카운트'], palette="Blues_d", ax=ax)

            # 막대 위에 비율 표시
            for index, value in enumerate(address_count_df['카운트']):
                ax.text(index, value + 10, f"{address_count_df['비율 (%)'].iloc[index]:.2f}%", ha='center')

            ax.set_title(f'{selected_branch} 지점 부산 고객 거주지 분포 및 비율')
            ax.set_xlabel('거주지')
            ax.set_ylabel('고객 수')
            ax.set_xticks(range(len(address_count_df)))
            ax.set_xticklabels(address_count_df['거주지'], rotation=45)

            # 그래프를 Streamlit에 표시
            st.pyplot(fig)

            st.dataframe(address_count_df, use_container_width=True)


        with tab3:
            st.header("폭스바겐 고객성별&연령")
            #st.write("여기에 아우디 페이지 내용을 추가하세요.")

        # 성별과 연령을 분리
            df[['성별', '연령']] = df['성별/연령'].str.split('/', expand=True)

            # 성별 데이터를 확인하기 위한 고유 값 출력
            #st.header("고객 성별 데이터 종류")
            #unique_genders = df['성별'].unique()
            #st.write(f"성별 데이터에 포함된 값들: {unique_genders}")

            # 성별 데이터를 통일
            df['성별'] = df['성별'].str.strip()  # 공백 제거
            df['성별'] = df['성별'].replace({
                '남자': '남성',
                '남': '남성',
                '남성': '남성',
                '여자': '여성',
                '여': '여성',
                '여성': '여성'
            })

            # 불필요한 값들을 제거 (유효한 값은 '남성'과 '여성'만)
            df = df[df['성별'].isin(['남성', '여성'])]
                        
            # 연령을 숫자로 변환
            df['연령'] = pd.to_numeric(df['연령'], errors='coerce')
            
            # 지점 선택을 위한 드롭다운 메뉴 (고유한 key 부여)
            selected_branch3 = st.selectbox('지점을 선택하세요:', df['지점명'].unique(), key="selectbox3")
            
            # 선택된 지점에 해당하는 데이터 필터링
            selected_df = df[df['지점명'] == selected_branch3]
            
            # 성별 및 연령대별로 카운트
            gender_count = selected_df['성별'].value_counts(normalize=True) * 100  # 성별 비율
            age_group_count = selected_df['연령'].value_counts(bins=[0, 20, 30, 40, 50,  60, 100], normalize=True) * 100  # 연령대 비율

            # 성별 비율 시각화
            st.subheader(f'{selected_branch3} 지점 고객 성별 비율')
            fig_gender, ax_gender = plt.subplots(figsize=(6, 4))
            sns.barplot(x=gender_count.index, y=gender_count.values, palette="Blues_d", ax=ax_gender)

            # 막대 위에 비율 텍스트 추가
            for i, value in enumerate(gender_count.values):
                ax_gender.text(i, value + 1, f'{value:.2f}%', ha='center')


            ax_gender.set_xlabel('성별')
            ax_gender.set_ylabel('비율 (%)')
            ax_gender.set_title(f'{selected_branch3} 지점 고객 성별 비율')
            st.pyplot(fig_gender)

            gender_df = pd.DataFrame({
                '성별': gender_count.index,
                '비율': [f'{value:.2f}%' for value in gender_count.values]
            })

            # 성별 비율 테이블 출력
            st.table(gender_df)

        


            # 연령대 구간 라벨 설정
            age_group_labels = ['0~20세', '20대', '30대', '40대', '50대', '60세 이상']

            # x축 라벨을 맞추기 위해 인덱스를 순서대로 설정
            age_group_count.index = pd.IntervalIndex(age_group_count.index)
            age_group_count = age_group_count.sort_index()  # 연령대를 어린 나이순으로 정렬

            # 연령대 비율 시각화
            st.subheader(f'{selected_branch3} 지점 고객 연령대 비율')
            fig_age, ax_age = plt.subplots(figsize=(6, 4))

            # 막대 그래프 그리기
            sns.barplot(x=age_group_labels, y=age_group_count.values, palette="Blues_d", ax=ax_age)

            # 막대 위에 비율 텍스트 추가
            for i, value in enumerate(age_group_count.values):
                ax_age.text(i, value + 1, f'{value:.2f}%', ha='center')

            ax_age.set_xlabel('연령대')
            ax_age.set_ylabel('비율 (%)')
            ax_age.set_title(f'{selected_branch3} 지점 고객 연령대 비율')

            # 그래프를 Streamlit에 표시
            st.pyplot(fig_age)

            # 연령대 비율 테이블 출력
            age_df = pd.DataFrame({
                '연령대': age_group_labels,
                '비율': [f'{value:.2f}%' for value in age_group_count.values]
            })
            st.table(age_df)


        with tab4:
            st.header("폭스바겐 지점별 판매차종")

            # 모델명에서 괄호와 그 안의 내용을 제거하는 함수
            def clean_model(model):
                model_without_brackets = re.sub(r'\(.*?\)', '', model).strip()
                return model_without_brackets.split()[0].strip()

            # 모델 데이터를 가공하여 괄호 내용 제거
            df['모델'] = df['모델'].apply(clean_model)

            # 전체 지점 판매대수 보기 버튼을 상단에 배치
            st.write("") 
            if st.button('전체 지점 모델 판매대수 보기', key="all_branches"):
                model_count_total = df['모델'].value_counts()  # 전체 지점의 모델별 판매대수
                total_sales = model_count_total.sum()  # 전체 판매대수
                model_percentage_total = (model_count_total / total_sales) * 100  # 전체 지점의 모델별 판매 비율

                st.subheader('전체 지점 모델별 판매 비율')
                fig_total_model, ax_total_model = plt.subplots(figsize=(6, 4))
                sns.barplot(x=model_count_total.index, y=model_percentage_total.values, palette="Blues_d", ax=ax_total_model)

                # 막대 위에 비율 텍스트 추가
                for i, value in enumerate(model_percentage_total.values):
                    ax_total_model.text(i, value + 1, f'{value:.2f}%', ha='center')

                ax_total_model.set_xlabel('모델')
                ax_total_model.set_ylabel('판매 비율 (%)')
                ax_total_model.set_title('전체 지점 모델별 판매 비율')
                st.pyplot(fig_total_model)

                # 전체 지점 모델별 판매대수 및 비율 테이블 출력
                model_df_total = pd.DataFrame({
                    '모델': model_count_total.index,
                    '판매대수': [f'{int(value)}대' for value in model_count_total.values],
                    '비율 (%)': [f'{value:.2f}%' for value in model_percentage_total.values]
                })

                st.dataframe(model_df_total, use_container_width=True)

            st.write("")  # 공간 추가

            # 지점 선택을 위한 드롭다운 메뉴 (고유한 key 부여)
            selected_branch4 = st.selectbox('지점을 선택하세요:', df['지점명'].unique(), key="selectbox5")

            # 선택한 지점에 해당하는 데이터 필터링
            filtered_df = df[df['지점명'] == selected_branch4]

            # 모델별 판매대수 및 비율 계산
            model_count_filtered = filtered_df['모델'].value_counts()
            total_sales_filtered = model_count_filtered.sum()
            model_percentage_filtered = (model_count_filtered / total_sales_filtered) * 100

            # 모델별 판매대수 시각화 (비율로 변경)
            st.subheader(f'{selected_branch4} 지점 모델별 판매 비율')
            fig_model, ax_model = plt.subplots(figsize=(6, 4))
            sns.barplot(x=model_count_filtered.index, y=model_percentage_filtered.values, palette="Blues_d", ax=ax_model)

            # 막대 위에 비율 텍스트 추가
            for i, value in enumerate(model_percentage_filtered.values):
                ax_model.text(i, value + 1, f'{value:.2f}%', ha='center')

            ax_model.set_xlabel('모델')
            ax_model.set_ylabel('판매 비율 (%)')
            ax_model.set_title(f'{selected_branch4} 지점 모델별 판매 비율')
            st.pyplot(fig_model)

            # 모델별 판매대수 및 비율 테이블 출력
            model_df_filtered = pd.DataFrame({
                '모델': model_count_filtered.index,
                '판매대수': [f'{int(value)}대' for value in model_count_filtered.values],
                '비율 (%)': [f'{value:.2f}%' for value in model_percentage_filtered.values]
            })

            # 모델별 판매대수 데이터프레임 출력
            st.dataframe(model_df_filtered, use_container_width=True)

            st.write("")  # 공간 추가